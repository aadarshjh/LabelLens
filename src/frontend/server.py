"""Small threaded HTTP server for the local LabelLens frontend."""

import http.server
import json
import os
import re
import socketserver
import tempfile
import threading
from email import policy
from email.parser import BytesParser

import cv2
import numpy as np

from src.config import FRONTEND_DIR
from src.pipeline.pipeline import run_end_to_end_pipeline, get_shared_retriever
from src.frontend.serializer import serialize_for_frontend

PORT = int(os.environ.get("PORT", "8000"))
API_KEY = os.environ.get("GROQ_API_KEY", "")
# Includes multipart framing. Override with MAX_REQUEST_BODY_BYTES when needed.
MAX_REQUEST_BODY_BYTES = int(os.environ.get("MAX_REQUEST_BODY_BYTES", str(12 * 1024 * 1024)))
if MAX_REQUEST_BODY_BYTES <= 0:
    raise ValueError("MAX_REQUEST_BODY_BYTES must be a positive integer")
ANALYSIS_SLOT = threading.BoundedSemaphore(1)


class Handler(http.server.SimpleHTTPRequestHandler):
    protocol_version = "HTTP/1.0"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(FRONTEND_DIR), **kwargs)

    def setup(self):
        super().setup()
        self.connection.settimeout(15)

    def end_headers(self):
        self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate, max-age=0')
        self.send_header('Pragma', 'no-cache')
        self.send_header('Expires', '0')
        super().end_headers()

    def _send_json(self, status, payload, extra_headers=None):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        for name, value in (extra_headers or {}).items():
            self.send_header(name, value)
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        if self.path != '/analyze':
            self.send_error(404, "Not Found")
            return

        tmp_path = None
        acquired = False
        try:
            # Reject transfer codings instead of allowing an unbounded/ambiguous body.
            if self.headers.get("Transfer-Encoding"):
                self._send_json(400, {"error": True, "message": "Transfer-Encoding is not supported."})
                return
            lengths = self.headers.get_all("Content-Length", [])
            if len(lengths) != 1 or not re.fullmatch(r"[0-9]+", lengths[0].strip()):
                self._send_json(400, {"error": True, "message": "Invalid Content-Length."})
                return
            try:
                content_length = int(lengths[0])
            except ValueError:
                self._send_json(400, {"error": True, "message": "Invalid Content-Length."})
                return
            if content_length > MAX_REQUEST_BODY_BYTES:
                self._send_json(413, {"error": True, "message": "Upload is too large."})
                return
            content_type = self.headers.get('Content-Type', '')
            if not content_type.lower().startswith('multipart/form-data'):
                self._send_json(400, {"error": True, "message": "Expected multipart/form-data."})
                return

            # At most one request can hold or read an upload for expensive analysis.
            if not ANALYSIS_SLOT.acquire(blocking=False):
                self._send_json(503, {"error": True, "message": "Another label is being analyzed. Please retry shortly."},
                                {"Retry-After": "5"})
                return
            acquired = True

            body = self.rfile.read(content_length)
            if len(body) != content_length:
                self._send_json(400, {"error": True, "message": "Incomplete upload request."})
                return

            # Use the standard MIME parser so boundaries, quoting, and part headers
            # are interpreted according to the multipart format.
            message = BytesParser(policy=policy.default).parsebytes(
                b"Content-Type: " + content_type.encode("latin-1") + b"\r\nMIME-Version: 1.0\r\n\r\n" + body
            )
            boundary = message.get_boundary()
            try:
                closing_boundary = b"--" + boundary.encode("ascii") + b"--" if boundary else None
            except UnicodeEncodeError:
                closing_boundary = None
            if (not message.is_multipart() or not closing_boundary
                    or not body.rstrip().endswith(closing_boundary)):
                self._send_json(400, {"error": True, "message": "Malformed upload request."})
                return
            image_data = None
            for part in message.iter_parts():
                if part.get_filename():
                    image_data = part.get_payload(decode=True)
                    break
            if not image_data:
                self._send_json(400, {"error": True, "message": "No image was included in the upload."})
                return

            decoded_image = cv2.imdecode(np.frombuffer(image_data, dtype=np.uint8), cv2.IMREAD_COLOR)
            if decoded_image is None or decoded_image.size == 0:
                self._send_json(400, {"error": True, "message": "The uploaded file is not a readable image."})
                return

            import uuid
            from src.audit_logger import STAGE_REQUEST_RECEIVED, audit_log, set_correlation_id
            from src.config import load_env

            load_env()
            corr_id = f"req_{uuid.uuid4().hex[:8]}"
            set_correlation_id(corr_id)
            audit_log(STAGE_REQUEST_RECEIVED, f"POST /analyze payload received ({len(image_data)} bytes)", correlation_id=corr_id)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
                tmp_file.write(image_data)
                tmp_path = tmp_file.name

            current_api_key = os.environ.get("GROQ_API_KEY", API_KEY)
            phase4_results, phase2_summary = run_end_to_end_pipeline(
                tmp_path, current_api_key, batch=True, correlation_id=corr_id,
                retriever=get_shared_retriever()
            )
            self._send_json(200, serialize_for_frontend(phase4_results, phase2_summary, "Scanned Product"))
        except (TimeoutError, ConnectionError):
            self._send_json(400, {"error": True, "message": "Incomplete upload request."})
        except ValueError as e:
            msg = str(e)
            if "not a readable image" in msg.lower() or "could not read image" in msg.lower() or "corrupted" in msg.lower():
                user_msg = "The uploaded file is not a readable image."
            else:
                user_msg = msg or "The uploaded image could not be processed."
            self._send_json(400, {"error": True, "message": user_msg})
        except Exception as e:
            print(f"Analysis request failed: {type(e).__name__}")
            self._send_json(500, {"error": True, "message": "We couldn't analyze this label. Please try again."})
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)
            if acquired:
                ANALYSIS_SLOT.release()


class ThreadingHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    """Serve static requests with bounded handlers and one bounded analysis."""
    daemon_threads = True
    block_on_close = False
    request_queue_size = 32
    max_active_handlers = 32

    def __init__(self, *args, **kwargs):
        self._handler_slots = threading.BoundedSemaphore(self.max_active_handlers)
        super().__init__(*args, **kwargs)

    def process_request(self, request, client_address):
        if not self._handler_slots.acquire(blocking=False):
            try:
                request.settimeout(0.25)
                request.sendall(
                    b"HTTP/1.0 503 Service Unavailable\r\nConnection: close\r\n"
                    b"Content-Length: 0\r\nRetry-After: 1\r\n\r\n"
                )
            except OSError:
                pass
            finally:
                self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except Exception:
            self._handler_slots.release()
            raise

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            self._handler_slots.release()


def run():
    ThreadingHTTPServer.allow_reuse_address = True
    try:
        get_shared_retriever()
        print("✓ Knowledge-base retriever initialized.")
    except Exception as exc:
        print(f"Notice: Retriever could not be pre-initialized: {exc}")

    with ThreadingHTTPServer(("", PORT), Handler) as httpd:
        print(f"Serving at port {PORT}")
        httpd.serve_forever()


if __name__ == '__main__':
    run()
