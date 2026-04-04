from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class JsonSessionStore:
    SUPPORTED_MODES = ("chat", "shock")
    SUPPORTED_PERSONAS = ("jekyllhyde", "femwife")

    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir
        self.sessions_dir = data_dir / "sessions"
        self.memories_dir = data_dir / "memories"
        self.logs_dir = data_dir / "logs"
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self.memories_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self._ensure_global_log_files()

    def _session_path(self, session_id: str) -> Path:
        return self.sessions_dir / f"{session_id}.json"

    def _memory_path(self, session_id: str, persona_key: str) -> Path:
        return self.memories_dir / f"{session_id}_{persona_key}.json"

    def _global_log_path(self, mode: str, persona_key: str) -> Path:
        return self.logs_dir / f"global_{mode}_{persona_key}.jsonl"

    def _write_json(self, path: Path, payload: dict[str, Any]) -> None:
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")

    def _read_json(self, path: Path) -> dict[str, Any]:
        return json.loads(path.read_text(encoding="utf-8"))

    def _append_jsonl(self, path: Path, payload: dict[str, Any]) -> None:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=True) + "\n")

    def _ensure_global_log_files(self) -> None:
        for mode in self.SUPPORTED_MODES:
            for persona_key in self.SUPPORTED_PERSONAS:
                path = self._global_log_path(mode, persona_key)
                if not path.exists():
                    path.write_text("", encoding="utf-8")

    def _normalize_mode(self, raw_mode: str | None, session_id: str) -> str:
        mode = (raw_mode or "").strip().lower()
        if mode in self.SUPPORTED_MODES:
            return mode

        prefix = session_id.strip().split("_", 1)[0].lower()
        if prefix in self.SUPPORTED_MODES:
            return prefix
        return ""

    def _normalize_persona(self, raw_persona: str | None) -> str:
        persona = (raw_persona or "").strip().lower()
        if persona in self.SUPPORTED_PERSONAS:
            return persona
        return ""

    def _append_global_log(
        self,
        *,
        session_id: str,
        mode: str | None,
        persona_key: str | None,
        event_type: str,
        payload: dict[str, Any],
    ) -> None:
        mode_key = self._normalize_mode(mode, session_id)
        persona = self._normalize_persona(persona_key)
        if not mode_key or not persona:
            return

        entry = {
            "timestamp": utc_now_iso(),
            "session_id": session_id,
            "mode": mode_key,
            "persona": persona,
            "event_type": event_type,
            **payload,
        }
        self._append_jsonl(self._global_log_path(mode_key, persona), entry)

    def create_session(self, payload: dict[str, Any]) -> dict[str, Any]:
        session_id = str(payload["session_id"])
        path = self._session_path(session_id)
        if path.exists():
            raise RuntimeError(f"Session already exists: {session_id}")
        self._write_json(path, payload)
        return payload

    def load_session(self, session_id: str) -> dict[str, Any]:
        path = self._session_path(session_id)
        if not path.exists():
            raise FileNotFoundError(f"Session not found: {session_id}")
        return self._read_json(path)

    def save_session(self, session: dict[str, Any]) -> None:
        session["updated_at"] = utc_now_iso()
        self._write_json(self._session_path(str(session["session_id"])), session)

    def append_transcript(self, session_id: str, event: dict[str, Any]) -> int:
        session = self.load_session(session_id)
        transcript = session.setdefault("transcript", [])
        event["index"] = len(transcript) + 1
        event.setdefault("timestamp", utc_now_iso())
        transcript.append(event)
        self.save_session(session)

        if str(event.get("type", "")).strip() == "user_input":
            participants = [self._normalize_persona(str(p)) for p in session.get("participants", [])]
            unique_participants = [p for p in dict.fromkeys(participants) if p]
            for persona_key in unique_participants:
                self._append_global_log(
                    session_id=session_id,
                    mode=str(event.get("mode") or session.get("mode") or ""),
                    persona_key=persona_key,
                    event_type="user_input",
                    payload={
                        "index": int(event["index"]),
                        "message": str(event.get("message", "")),
                        "source": "transcript",
                    },
                )
        return int(event["index"])

    def ensure_persona_memory_file(self, session_id: str, persona_key: str, mode: str) -> None:
        path = self._memory_path(session_id, persona_key)
        if path.exists():
            return

        payload = {
            "session_id": session_id,
            "persona": persona_key,
            "mode": mode,
            "created_at": utc_now_iso(),
            "entries": [],
        }
        self._write_json(path, payload)

    def append_persona_memory(
        self,
        session_id: str,
        persona_key: str,
        kind: str,
        content: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        path = self._memory_path(session_id, persona_key)
        if not path.exists():
            raise FileNotFoundError(f"Memory file not found for session/persona: {session_id}/{persona_key}")

        payload = self._read_json(path)
        entries = payload.setdefault("entries", [])
        memory_id = f"{persona_key}-m-{len(entries) + 1}"

        entry = {
            "memory_id": memory_id,
            "timestamp": utc_now_iso(),
            "kind": kind,
            "content": content,
            "metadata": metadata or {},
        }
        entries.append(entry)
        self._write_json(path, payload)
        self._append_global_log(
            session_id=session_id,
            mode=str(payload.get("mode", "")),
            persona_key=persona_key,
            event_type="persona_output",
            payload={
                "memory_id": memory_id,
                "kind": kind,
                "content": content,
                "metadata": metadata or {},
            },
        )
        return memory_id

    def get_persona_memories(self, session_id: str, persona_key: str) -> list[dict[str, Any]]:
        path = self._memory_path(session_id, persona_key)
        if not path.exists():
            return []
        payload = self._read_json(path)
        return list(payload.get("entries", []))

    def list_session_memory_files(self, session_id: str) -> list[str]:
        files = sorted(self.memories_dir.glob(f"{session_id}_*.json"))
        return [str(path) for path in files]
