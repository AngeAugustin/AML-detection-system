from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable
from urllib.parse import urlencode, urljoin
from urllib.request import Request, urlopen


@dataclass(frozen=True)
class ExternalIngestionConfig:
    base_url: str
    endpoint: str = "/transactions"
    auth_token: str = ""
    poll_interval_seconds: float = 5.0
    limit: int = 200
    timeout_seconds: float = 15.0
    cursor_param: str = "cursor"
    limit_param: str = "limit"
    initial_cursor: str = ""


class ExternalTransactionIngestor:
    """Polling ingestor for an external transactions API."""

    def __init__(self, config: ExternalIngestionConfig, on_batch: Callable[[list[dict[str, Any]]], None]) -> None:
        self._config = config
        self._on_batch = on_batch
        self._cursor = config.initial_cursor
        self._running = False
        self._thread: threading.Thread | None = None
        self._last_error = ""
        self._fetched_total = 0
        self._processed_total = 0
        self._last_poll_at = 0.0
        self._lock = threading.Lock()

    def start(self) -> None:
        with self._lock:
            if self._running:
                return
            self._running = True
            self._thread = threading.Thread(target=self._run_loop, daemon=True)
            self._thread.start()

    def stop(self) -> None:
        with self._lock:
            self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "running": self._running,
                "cursor": self._cursor,
                "last_error": self._last_error,
                "fetched_total": self._fetched_total,
                "processed_total": self._processed_total,
                "last_poll_at": self._last_poll_at,
                "endpoint": self._config.endpoint,
                "base_url": self._config.base_url,
            }

    def _run_loop(self) -> None:
        while True:
            with self._lock:
                if not self._running:
                    return
            try:
                batch, next_cursor = self._fetch_once()
                with self._lock:
                    self._fetched_total += len(batch)
                    self._last_poll_at = time.time()
                    self._cursor = next_cursor
                    self._last_error = ""
                if batch:
                    self._on_batch(batch)
                    with self._lock:
                        self._processed_total += len(batch)
            except Exception as e:  # noqa: BLE001
                with self._lock:
                    self._last_error = str(e)
            time.sleep(max(0.2, self._config.poll_interval_seconds))

    def _fetch_once(self) -> tuple[list[dict[str, Any]], str]:
        q: dict[str, str | int] = {self._config.limit_param: self._config.limit}
        if self._cursor:
            q[self._config.cursor_param] = self._cursor
        url = urljoin(self._config.base_url.rstrip("/") + "/", self._config.endpoint.lstrip("/"))
        full_url = f"{url}?{urlencode(q)}"

        headers = {"Accept": "application/json"}
        if self._config.auth_token:
            headers["Authorization"] = f"Bearer {self._config.auth_token}"
        req = Request(url=full_url, headers=headers, method="GET")
        with urlopen(req, timeout=self._config.timeout_seconds) as resp:  # noqa: S310
            payload = json.loads(resp.read().decode("utf-8"))

        if isinstance(payload, list):
            return [x for x in payload if isinstance(x, dict)], self._cursor

        if not isinstance(payload, dict):
            raise ValueError("External API response must be object or list")
        tx = payload.get("transactions", [])
        if not isinstance(tx, list):
            raise ValueError("External API field 'transactions' must be a list")
        batch = [x for x in tx if isinstance(x, dict)]
        next_cursor = str(payload.get("next_cursor", self._cursor))
        return batch, next_cursor
