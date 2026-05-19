from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


@dataclass
class _PayloadStore:
    lock: threading.Lock = field(default_factory=threading.Lock)
    payload: dict[str, Any] | None = None
    received_at: datetime | None = None

    def set_payload(self, payload: dict[str, Any]) -> None:
        with self.lock:
            self.payload = payload
            self.received_at = datetime.now()

    def get_payload(self) -> dict[str, Any] | None:
        with self.lock:
            if self.payload is None:
                return None
            return dict(self.payload)


class GSIServer:
    def __init__(self, host: str = '127.0.0.1', port: int = 3000) -> None:
        self.host = host
        self.port = port
        self._store = _PayloadStore()
        self._httpd: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        store = self._store
        logger = logging.getLogger(__name__)

        class GSIRequestHandler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802
                content_length = int(self.headers.get('Content-Length', '0'))
                body = self.rfile.read(content_length)
                try:
                    payload = json.loads(body.decode('utf-8'))
                except json.JSONDecodeError:
                    self.send_response(400)
                    self.end_headers()
                    self.wfile.write(b'invalid json')
                    return

                if not isinstance(payload, dict):
                    self.send_response(400)
                    self.end_headers()
                    self.wfile.write(b'json payload must be object')
                    return

                store.set_payload(payload)
                logger.info(
                    'GSI payload received at %s | keys=%s | player=%s | allplayers=%s',
                    store.received_at.strftime('%H:%M:%S') if store.received_at else 'unknown',
                    sorted(payload.keys()),
                    'player' in payload,
                    'allplayers' in payload,
                )
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b'ok')

            def do_GET(self) -> None:  # noqa: N802
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b'cs2-ai-sandbox gsi server')

            def log_message(self, format: str, *args: object) -> None:
                return

        self._httpd = ThreadingHTTPServer((self.host, self.port), GSIRequestHandler)
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True, name='gsi-server-thread')
        self._thread.start()
        logger.info('GSI server listening on http://%s:%s', self.host, self.port)

    def stop(self) -> None:
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd = None
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

    def get_latest_payload(self) -> dict[str, Any] | None:
        return self._store.get_payload()
