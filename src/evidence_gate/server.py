"""A loopback-only, read-only viewer for a single saved experiment report.

The report is validated and snapshotted at startup. Request paths can never
select a local file; only the three bundled assets and two API routes exist.
"""

from __future__ import annotations

import ipaddress
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

MAX_REPORT_BYTES = 32 * 1024 * 1024
WEB_ROOT = Path(__file__).resolve().parent / "web"
STATIC_ROUTES = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/index.html": ("index.html", "text/html; charset=utf-8"),
    "/style.css": ("style.css", "text/css; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
}


def load_report(report_path: str | Path) -> dict[str, Any]:
    """Read one bounded JSON document without accepting request-time paths."""
    path = Path(report_path).expanduser().resolve(strict=True)
    if not path.is_file():
        raise ValueError("Report must be a JSON file")
    with path.open("rb") as stream:
        raw = stream.read(MAX_REPORT_BYTES + 1)
    if len(raw) > MAX_REPORT_BYTES:
        raise ValueError("Report exceeds the 32 MiB viewer limit")
    try:
        report = json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError) as exc:
        raise ValueError("Report is not valid UTF-8 JSON") from exc
    if not isinstance(report, dict) or not isinstance(report.get("methods"), list):
        raise ValueError("Report must be an object with a methods list")
    if any(not isinstance(method, dict) for method in report["methods"]):
        raise ValueError("Each report method must be an object")
    return report


def create_server(
    report_path: str | Path,
    host: str = "127.0.0.1",
    port: int = 8766,
) -> ThreadingHTTPServer:
    """Build the server; port=0 is useful for local tests.

    Public binding is intentionally unsupported: this viewer has no login and
    is meant for an individual's local experiment artifacts.
    """
    if host == "localhost":
        host = "127.0.0.1"
    try:
        address = ipaddress.ip_address(host)
    except ValueError as exc:
        raise ValueError("Viewer host must be an IPv4 loopback address") from exc
    if address.version != 4 or not address.is_loopback:
        raise ValueError("Viewer host must be an IPv4 loopback address")

    report = load_report(report_path)
    payload = json.dumps(report, ensure_ascii=False, allow_nan=False).encode("utf-8")
    assets = {
        route: ((WEB_ROOT / filename).read_bytes(), content_type)
        for route, (filename, content_type) in STATIC_ROUTES.items()
    }

    class ReportHandler(BaseHTTPRequestHandler):
        server_version = "EvidenceGate"
        sys_version = ""

        def log_message(self, format: str, *args: Any) -> None:
            # Avoid logging request-supplied paths/headers or report contents.
            return

        def _send(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Frame-Options", "DENY")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'none'; script-src 'self'; style-src 'self'; "
                "connect-src 'self'; img-src 'self'; font-src 'self'; "
                "base-uri 'none'; form-action 'none'; frame-ancestors 'none'",
            )
            self.send_header("Permissions-Policy", "camera=(), microphone=(), geolocation=()")
            if status == 405:
                self.send_header("Allow", "GET, HEAD")
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)

        def _error(self, status: int, message: str) -> None:
            self.close_connection = True
            self._send(status, json.dumps({"error": message}).encode(), "application/json")

        def _read(self) -> None:
            port_ = self.server.server_port
            allowed_hosts = {f"{host}:{port_}", f"localhost:{port_}"}
            if port_ == 80:
                allowed_hosts.update({host, "localhost"})
            if self.headers.get("Host", "") not in allowed_hosts:
                self._error(403, "Invalid host")
                return
            origin = self.headers.get("Origin")
            if origin and origin not in {f"http://{name}" for name in allowed_hosts}:
                self._error(403, "Cross-origin access is disabled")
                return
            if self.headers.get("Sec-Fetch-Site") == "cross-site":
                self._error(403, "Cross-site access is disabled")
                return
            if self.headers.get("Transfer-Encoding") or self.headers.get("Content-Length", "0") != "0":
                self._error(400, "Request bodies are not accepted")
                return
            parsed = urlsplit(self.path)
            if parsed.scheme or parsed.netloc or parsed.query or parsed.fragment:
                self._error(404, "Route not found")
                return
            if self.path == "/api/health":
                self._send(200, b'{"status":"ok","read_only":true}', "application/json")
            elif self.path == "/api/report":
                self._send(200, payload, "application/json; charset=utf-8")
            elif self.path in assets:
                body, content_type = assets[self.path]
                self._send(200, body, content_type)
            else:
                self._error(404, "Route not found")

        do_GET = _read
        do_HEAD = _read

        def _reject_write(self) -> None:
            self._error(405, "This viewer is read-only")

        do_POST = _reject_write
        do_PUT = _reject_write
        do_PATCH = _reject_write
        do_DELETE = _reject_write
        do_OPTIONS = _reject_write
        do_TRACE = _reject_write
        do_CONNECT = _reject_write

    server = ThreadingHTTPServer((host, port), ReportHandler)
    server.daemon_threads = True
    return server


def serve(report_path: str | Path, host: str = "127.0.0.1", port: int = 8766) -> None:
    """Serve a saved report until interrupted; never run or modify experiments."""
    server = create_server(report_path, host, port)
    print(f"EvidenceGate: http://{server.server_address[0]}:{server.server_port}", flush=True)
    print("Read-only report snapshot. Press Ctrl+C to stop.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
