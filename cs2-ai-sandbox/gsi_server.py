from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
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
        self._dump_path = Path(__file__).resolve().parent / 'latest_gsi_payload.json'

    def _write_debug_dump(self, payload: dict[str, Any]) -> None:
        self._dump_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding='utf-8',
        )

    def _log_payload_summary(self, payload: dict[str, Any], logger: logging.Logger) -> None:
        player_block = payload.get('player')
        allplayers_block = payload.get('allplayers')
        player_position = player_block.get('position') if isinstance(player_block, dict) else None
        player_forward = player_block.get('forward') if isinstance(player_block, dict) else None
        player_state = player_block.get('state') if isinstance(player_block, dict) else None
        logger.info(
            'GSI payload received at %s | keys=%s | player=%s | allplayers=%s',
            self._store.received_at.strftime('%H:%M:%S') if self._store.received_at else 'unknown',
            sorted(payload.keys()),
            isinstance(player_block, dict),
            isinstance(allplayers_block, dict),
        )
        logger.info(
            'GSI payload details | player_position=%r | player_forward=%r | player_state_keys=%s | allplayers_count=%s | dump=%s',
            player_position,
            player_forward,
            sorted(player_state.keys()) if isinstance(player_state, dict) else None,
            len(allplayers_block) if isinstance(allplayers_block, dict) else 0,
            self._dump_path,
        )

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
                try:
                    self.server.gsi_server._write_debug_dump(payload)  # type: ignore[attr-defined]
                except Exception as exc:
                    logger.warning('Failed to write GSI payload dump: %s', exc)
                self.server.gsi_server._log_payload_summary(payload, logger)  # type: ignore[attr-defined]
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
        self._httpd.gsi_server = self  # type: ignore[attr-defined]
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
