"""HTTP boundary and concurrency regression tests for the frontend server."""
import http.client
import json
import os
import threading
import time
from contextlib import contextmanager

import pytest

cv2 = pytest.importorskip("cv2")
import numpy as np
from src.frontend import server


@contextmanager
def running_server():
    httpd = server.ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield httpd.server_address[1]
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=2)


def image_multipart():
    ok, encoded = cv2.imencode(".png", np.zeros((2, 2, 3), dtype=np.uint8))
    assert ok
    boundary = b"test-boundary"
    data = (b"--" + boundary + b"\r\nContent-Disposition: form-data; name=\"file\"; filename=\"label.png\"\r\n"
            b"Content-Type: image/png\r\n\r\n" + encoded.tobytes() + b"\r\n--" + boundary + b"--\r\n")
    return boundary, data


def post_raw(port, body, content_length=None):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    conn.putrequest("POST", "/analyze")
    conn.putheader("Content-Type", "multipart/form-data; boundary=test-boundary")
    if content_length is not None:
        conn.putheader("Content-Length", content_length)
    else:
        conn.putheader("Content-Length", str(len(body)))
    conn.endheaders(body)
    response = conn.getresponse()
    result = response.status, response.read()
    conn.close()
    return result


def post_headers_and_partial_body(port, body, partial):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    conn.putrequest("POST", "/analyze")
    conn.putheader("Content-Type", "multipart/form-data; boundary=test-boundary")
    conn.putheader("Content-Length", str(len(body)))
    conn.endheaders(partial)
    response = conn.getresponse()
    result = response.status, response.read()
    conn.close()
    return result


def wait_for_analysis_slot(timeout=3):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if server.ANALYSIS_SLOT.acquire(blocking=False):
            server.ANALYSIS_SLOT.release()
            return True
        time.sleep(0.01)
    return False


def test_valid_and_exact_limit_uploads_call_pipeline(monkeypatch):
    _, body = image_multipart()
    calls = []
    monkeypatch.setattr(server, "run_end_to_end_pipeline", lambda *a, **k: ([], {"total": 0}))
    monkeypatch.setattr(server, "serialize_for_frontend", lambda *a: calls.append(True) or {"ok": True})
    monkeypatch.setattr(server, "MAX_REQUEST_BODY_BYTES", len(body))
    with running_server() as port:
        status, _ = post_raw(port, body)
    assert status == 200
    assert calls == [True]


def test_oversize_and_malformed_requests_do_not_call_pipeline(monkeypatch):
    _, body = image_multipart()
    calls = []
    monkeypatch.setattr(server, "run_end_to_end_pipeline", lambda *a, **k: calls.append(True))
    monkeypatch.setattr(server, "MAX_REQUEST_BODY_BYTES", len(body) - 1)
    with running_server() as port:
        assert post_raw(port, body)[0] == 413
        monkeypatch.setattr(server, "MAX_REQUEST_BODY_BYTES", len(body) + 10)
        assert post_raw(port, body[:-8])[0] == 400
        assert post_raw(port, body, "-1")[0] == 400
        assert post_raw(port, body, "9" * 5000)[0] == 400
    assert calls == []


def test_analysis_does_not_block_static_request_and_second_analysis_is_bounded(monkeypatch):
    _, body = image_multipart()
    entered = threading.Event()
    release = threading.Event()
    def slow_pipeline(*args, **kwargs):
        entered.set()
        release.wait(3)
        return [], {"total": 0}
    monkeypatch.setattr(server, "run_end_to_end_pipeline", slow_pipeline)
    monkeypatch.setattr(server, "serialize_for_frontend", lambda *a: {"ok": True})
    with running_server() as port:
        first_result = []
        first = threading.Thread(target=lambda: first_result.append(post_raw(port, body)[0]))
        first.start()
        assert entered.wait(2)
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
        conn.request("GET", "/history.html")
        static_response = conn.getresponse()
        assert static_response.status == 200
        static_response.read()
        conn.close()
        assert post_raw(port, body)[0] == 503
        release.set()
        first.join(3)
        assert first_result == [200]


def test_server_caps_active_handlers():
    httpd = server.ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
    for _ in range(httpd.max_active_handlers):
        assert httpd._handler_slots.acquire(blocking=False)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        conn = http.client.HTTPConnection("127.0.0.1", httpd.server_address[1], timeout=2)
        conn.request("GET", "/history.html")
        response = conn.getresponse()
        assert response.status == 503
        assert response.getheader("Retry-After") == "1"
        response.read()
        conn.close()
    finally:
        for _ in range(httpd.max_active_handlers):
            httpd._handler_slots.release()
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=2)


def test_analysis_exception_cleans_temp_file_and_releases_slot(monkeypatch):
    _, body = image_multipart()
    real_tempfile = server.tempfile.NamedTemporaryFile
    temp_paths = []
    calls = []

    def tracked_tempfile(*args, **kwargs):
        result = real_tempfile(*args, **kwargs)
        temp_paths.append(result.name)
        return result

    def fail_once(*args, **kwargs):
        calls.append(True)
        if len(calls) == 1:
            raise RuntimeError("test pipeline failure")
        return [], {"total": 0}

    monkeypatch.setattr(server.tempfile, "NamedTemporaryFile", tracked_tempfile)
    monkeypatch.setattr(server, "run_end_to_end_pipeline", fail_once)
    monkeypatch.setattr(server, "serialize_for_frontend", lambda *a: {"ok": True})
    with running_server() as port:
        assert post_raw(port, body)[0] == 500
        assert wait_for_analysis_slot()
        assert post_raw(port, body)[0] == 200
    assert len(calls) == 2
    assert all(not os.path.exists(path) for path in temp_paths)


def test_client_disconnect_releases_analysis_slot_and_cleans_temp_file(monkeypatch):
    _, body = image_multipart()
    entered = threading.Event()
    finish = threading.Event()
    real_tempfile = server.tempfile.NamedTemporaryFile
    temp_paths = []

    def tracked_tempfile(*args, **kwargs):
        result = real_tempfile(*args, **kwargs)
        temp_paths.append(result.name)
        return result

    def slow_pipeline(*args, **kwargs):
        entered.set()
        finish.wait(3)
        return [], {"total": 0}

    monkeypatch.setattr(server.tempfile, "NamedTemporaryFile", tracked_tempfile)
    monkeypatch.setattr(server, "run_end_to_end_pipeline", slow_pipeline)
    monkeypatch.setattr(server, "serialize_for_frontend", lambda *a: {"ok": True})
    with running_server() as port:
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        conn.putrequest("POST", "/analyze")
        conn.putheader("Content-Type", "multipart/form-data; boundary=test-boundary")
        conn.putheader("Content-Length", str(len(body)))
        conn.endheaders(body)
        assert entered.wait(2)
        conn.close()
        finish.set()
        assert wait_for_analysis_slot()
        assert post_raw(port, body)[0] == 200
    assert len(temp_paths) == 2
    assert all(not os.path.exists(path) for path in temp_paths)


def test_upload_timeout_releases_analysis_slot(monkeypatch):
    _, body = image_multipart()
    original_setup = server.Handler.setup
    calls = []

    def short_timeout_setup(handler):
        original_setup(handler)
        handler.connection.settimeout(0.1)

    monkeypatch.setattr(server.Handler, "setup", short_timeout_setup)
    monkeypatch.setattr(server, "run_end_to_end_pipeline", lambda *a, **k: calls.append(True) or ([], {"total": 0}))
    monkeypatch.setattr(server, "serialize_for_frontend", lambda *a: {"ok": True})
    with running_server() as port:
        assert post_headers_and_partial_body(port, body, body[:10])[0] in (400, 500)
        assert wait_for_analysis_slot()
        assert post_raw(port, body)[0] == 200
    assert calls == [True]


def test_invalid_image_upload_returns_400_and_server_remains_healthy(monkeypatch):
    calls = []
    monkeypatch.setattr(server, "run_end_to_end_pipeline", lambda *a, **k: calls.append(True) or ([], {"total": 0}))
    monkeypatch.setattr(server, "serialize_for_frontend", lambda *a: {"ok": True})

    boundary = b"test-boundary"
    invalid_data = (b"--" + boundary + b"\r\nContent-Disposition: form-data; name=\"file\"; filename=\"corrupt.png\"\r\n"
                    b"Content-Type: image/png\r\n\r\nnot-an-image-data\r\n--" + boundary + b"--\r\n")

    _, valid_body = image_multipart()

    with running_server() as port:
        status, resp_body = post_raw(port, invalid_data)
        assert status == 400
        data = json.loads(resp_body)
        assert data.get("error") is True
        assert "not a readable image" in data.get("message", "").lower()

        # Confirm server remains alive and subsequent valid request succeeds
        status_valid, _ = post_raw(port, valid_body)
        assert status_valid == 200

    assert calls == [True]


def test_repeated_analyses_reuse_retriever(monkeypatch):
    passed_retrievers = []

    def mock_pipeline(*args, **kwargs):
        passed_retrievers.append(kwargs.get("retriever"))
        return [], {"total": 0}

    monkeypatch.setattr(server, "run_end_to_end_pipeline", mock_pipeline)
    monkeypatch.setattr(server, "serialize_for_frontend", lambda *a: {"ok": True})

    _, body = image_multipart()
    with running_server() as port:
        assert post_raw(port, body)[0] == 200
        assert post_raw(port, body)[0] == 200

    assert len(passed_retrievers) == 2
    assert passed_retrievers[0] is not None
    assert passed_retrievers[0] is passed_retrievers[1]
