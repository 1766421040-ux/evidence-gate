"""HTTP boundary tests for the read-only local report viewer."""

from __future__ import annotations

import http.client
import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from evidence_gate.server import create_server, load_report


class ReportValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "report.json"

    def test_rejects_wrong_shape_and_invalid_json(self) -> None:
        for content in ['[]', '{"methods":{}}', '{"methods":[1]}', '{not-json']:
            with self.subTest(content=content):
                self.path.write_text(content)
                with self.assertRaises(ValueError):
                    load_report(self.path)

    def test_report_size_is_bounded_before_parsing(self) -> None:
        self.path.write_text('{"methods":[],"padding":"' + "x" * 100 + '"}')
        with patch("evidence_gate.server.MAX_REPORT_BYTES", 40):
            with self.assertRaisesRegex(ValueError, "limit"):
                load_report(self.path)

    def test_non_loopback_bind_is_rejected(self) -> None:
        self.path.write_text('{"methods":[]}')
        for host in ["0.0.0.0", "192.168.1.1", "example.com", "::"]:
            with self.subTest(host=host):
                with self.assertRaisesRegex(ValueError, "loopback"):
                    create_server(self.path, host, 0)


class ReportServerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "report.json"
        self.report = {
            "title": "真实回放 <script>alert(1)</script>",
            "methods": [{"name": "model", "kind": "model", "cases": []}],
        }
        self.path.write_text(json.dumps(self.report, ensure_ascii=False), encoding="utf-8")
        self.server = create_server(self.path, port=0)
        self.thread = threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
        self.thread.start()
        self.addCleanup(self._stop_server)

    def _stop_server(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def request(self, path: str, method: str = "GET", headers: dict | None = None, body: bytes | None = None):
        connection = http.client.HTTPConnection("127.0.0.1", self.server.server_port, timeout=2)
        try:
            connection.request(method, path, body=body, headers=headers or {})
            response = connection.getresponse()
            return response.status, dict(response.getheaders()), response.read()
        finally:
            connection.close()

    def test_report_is_utf8_json_and_read_only_snapshot(self) -> None:
        self.path.write_text('{"methods":[],"replaced":true}')
        status, headers, body = self.request("/api/report")
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body), self.report)
        self.assertIn("application/json", headers["Content-Type"])
        self.assertEqual(headers["Cache-Control"], "no-store")
        self.assertNotIn("Access-Control-Allow-Origin", headers)
        self.assertEqual(headers["X-Content-Type-Options"], "nosniff")

    def test_health_does_not_expose_file_path_or_report(self) -> None:
        status, _, body = self.request("/api/health")
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body), {"status": "ok", "read_only": True})

    def test_only_bundled_assets_are_served_with_csp(self) -> None:
        for path, mime in [("/", "text/html"), ("/index.html", "text/html"), ("/app.js", "text/javascript"), ("/style.css", "text/css")]:
            with self.subTest(path=path):
                status, headers, body = self.request(path)
                self.assertEqual(status, 200)
                self.assertIn(mime, headers["Content-Type"])
                self.assertGreater(len(body), 100)
                self.assertIn("script-src 'self'", headers["Content-Security-Policy"])
                self.assertIn("frame-ancestors 'none'", headers["Content-Security-Policy"])

    def test_head_has_headers_but_no_body(self) -> None:
        status, headers, body = self.request("/api/report", "HEAD")
        self.assertEqual(status, 200)
        self.assertGreater(int(headers["Content-Length"]), 0)
        self.assertEqual(body, b"")

    def test_query_and_traversal_cannot_select_local_files(self) -> None:
        paths = ["/../report.json", "/%2e%2e/report.json", "/api/report?path=/etc/passwd", "/api/report/", "/web/", "/etc/passwd", "//example.com/api/report"]
        for path in paths:
            with self.subTest(path=path):
                status, _, body = self.request(path)
                self.assertEqual(status, 404)
                self.assertNotIn(str(self.path).encode(), body)

    def test_writes_and_preflights_are_rejected(self) -> None:
        for method in ["POST", "PUT", "PATCH", "DELETE", "OPTIONS", "TRACE", "CONNECT"]:
            with self.subTest(method=method):
                status, headers, _ = self.request("/api/report", method)
                self.assertEqual(status, 405)
                self.assertEqual(headers["Allow"], "GET, HEAD")
        self.assertEqual(json.loads(self.path.read_text()), self.report)

    def test_cross_origin_requests_and_dns_rebinding_are_rejected(self) -> None:
        for headers in [{"Origin": "https://malicious.example"}, {"Origin": "null"}, {"Host": "malicious.example"}, {"Sec-Fetch-Site": "cross-site"}]:
            with self.subTest(headers=headers):
                self.assertEqual(self.request("/api/report", headers=headers)[0], 403)

    def test_same_origin_request_is_allowed(self) -> None:
        origin = f"http://127.0.0.1:{self.server.server_port}"
        self.assertEqual(self.request("/api/report", headers={"Origin": origin})[0], 200)

    def test_get_request_body_is_rejected(self) -> None:
        self.assertEqual(self.request("/api/report", body=b'{"path":"/etc/passwd"}')[0], 400)


if __name__ == "__main__":
    unittest.main()
