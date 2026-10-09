import http.server
import socketserver
import json
import os
import tempfile
from pathlib import Path
import cv2
import numpy as np
from src.config import FRONTEND_DIR
from src.pipeline.pipeline import run_end_to_end_pipeline
from src.frontend.serializer import serialize_for_frontend

PORT = int(os.environ.get("PORT", "8000"))
API_KEY = os.environ.get("GROQ_API_KEY", "")

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(FRONTEND_DIR), **kwargs)

    def end_headers(self):
        self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate, max-age=0')
        self.send_header('Pragma', 'no-cache')
        self.send_header('Expires', '0')
        super().end_headers()

    def _send_json(self, status, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        if self.path == '/analyze':
            tmp_path = None
            try:
                # Parse multipart/form-data
                content_type = self.headers.get('Content-Type')
                if not content_type or not content_type.startswith('multipart/form-data'):
                    self._send_json(400, {"error": True, "message": "Expected multipart/form-data."})
                    return

                # In Python 3, cgi.FieldStorage is deprecated but still works for basic multipart
                import email
                from email.message import EmailMessage

                # Fast standard library multipart parsing
                content_length = int(self.headers.get('Content-Length', 0))
                body = self.rfile.read(content_length)

                try:
                    boundary_parts = content_type.split("boundary=")
                    if len(boundary_parts) < 2:
                        self._send_json(400, {"error": True, "message": "Malformed upload request."})
                        return
                    boundary = boundary_parts[1].strip().strip('"').encode()
                    parts = body.split(b'--' + boundary)
                except Exception:
                    self._send_json(400, {"error": True, "message": "Malformed upload request."})
                    return

                image_data = None
                for part in parts:
                    if b'filename="' in part:
                        header_end = part.find(b'\r\n\r\n')
                        if header_end != -1:
                            # Safely extract avoiding IndexError if it's too short
                            candidate = part[header_end+4:]
                            if candidate.endswith(b'\r\n'):
                                image_data = candidate[:-2]
                            else:
                                image_data = candidate
                            break

                if not image_data:
                    self._send_json(400, {"error": True, "message": "No image was included in the upload."})
                    return

                decoded_image = cv2.imdecode(np.frombuffer(image_data, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
                if decoded_image is None or decoded_image.size == 0:
                    self._send_json(400, {"error": True, "message": "The uploaded file is not a readable image."})
                    return

                import uuid
                from src.audit_logger import (
                    STAGE_REQUEST_RECEIVED,
                    audit_log,
                    set_correlation_id,
                )
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
                    tmp_path, current_api_key, batch=True, correlation_id=corr_id
                )

                # Do not fabricate product names based on filename
                product_name = "Scanned Product"
                consumer_json = serialize_for_frontend(phase4_results, phase2_summary, product_name)

                self._send_json(200, consumer_json)

            except Exception as e:
                print(f"Analysis request failed: {type(e).__name__}")
                self._send_json(500, {
                    "error": True,
                    "message": "We couldn't analyze this label. Please try again."
                })
            finally:
                if tmp_path and os.path.exists(tmp_path):
                    os.remove(tmp_path)
        else:
            self.send_error(404, "Not Found")

def run():
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        print(f"Serving at port {PORT}")
        httpd.serve_forever()

if __name__ == '__main__':
    run()
