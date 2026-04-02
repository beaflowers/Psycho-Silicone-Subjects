"""Tiny JSON-backed store for one saved web chat state."""

from __future__ import annotations

import json
from pathlib import Path
from threading import Lock
from typing import Any


class ChatStateStore:
    """Persist a single current chat state to a local JSON file."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = Lock()
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> dict[str, Any] | None:
        with self._lock:
            if not self._path.exists():
                return None

            payload = json.loads(self._path.read_text(encoding="utf-8"))
            return dict(payload) if isinstance(payload, dict) else None

    def save(self, state: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            self._path.write_text(
                json.dumps(state, indent=2, ensure_ascii=True),
                encoding="utf-8",
            )
            return dict(state)
