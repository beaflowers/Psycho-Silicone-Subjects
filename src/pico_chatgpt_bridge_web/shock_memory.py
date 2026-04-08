"""Shock session and memory helpers for role-aware chat context."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
import re
from pathlib import Path
from typing import Any


SESSION_FILE_RE = re.compile(r"^shock_([a-f0-9]+)\.json$")
MEMORY_FILE_RE = re.compile(r"^shock_([a-f0-9]+)_(femwife|jekyllhyde)\.json$")
QUERY_TOKEN_RE = re.compile(r"[a-z0-9]{3,}")


def _normalise_session_key(value: str | None) -> str:
    if not value:
        return ""
    cleaned = str(value).strip().lower()
    if cleaned.startswith("shock_"):
        cleaned = cleaned[6:]
    return cleaned


def _session_id_from_key(key: str) -> str:
    return f"shock_{key}" if key else ""


def _read_json_dict(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _parse_timestamp(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    if not cleaned:
        return None
    try:
        if cleaned.endswith("Z"):
            cleaned = cleaned[:-1] + "+00:00"
        return datetime.fromisoformat(cleaned)
    except ValueError:
        return None


def _entry_excerpt(entry: dict[str, Any], max_chars: int = 420) -> str:
    raw_content = str(entry.get("content", "") or "").strip()
    if "\n\n{" in raw_content:
        raw_content = raw_content.split("\n\n{", 1)[0].strip()
    compact = " ".join(raw_content.split())
    if len(compact) <= max_chars:
        return compact
    return f"{compact[: max_chars - 3].rstrip()}..."


def _entry_score(entry: dict[str, Any], query_terms: set[str]) -> int:
    if not query_terms:
        return 0
    content = _entry_excerpt(entry, max_chars=1200).lower()
    kind = str(entry.get("kind", "") or "").lower()
    haystack_terms = set(QUERY_TOKEN_RE.findall(f"{kind} {content}"))
    return len(query_terms.intersection(haystack_terms))


def _sort_entries(entries: list[dict[str, Any]], query: str) -> list[dict[str, Any]]:
    query_terms = set(QUERY_TOKEN_RE.findall(query.lower()))

    def sort_key(entry: dict[str, Any]) -> tuple[int, datetime]:
        score = _entry_score(entry, query_terms)
        ts = _parse_timestamp(entry.get("timestamp")) or datetime.min
        return score, ts

    return sorted(entries, key=sort_key, reverse=True)


def _persona_identity_anchor(character_persona: str, role: str) -> str:
    if character_persona == "jekyllhyde":
        return (
            "Identity anchor: You are the Jekyll/Hyde subject in this archive. "
            "Stay in-character at all times and never answer like a generic assistant."
        )
    if character_persona == "femwife":
        return (
            "Identity anchor: You are the Angela/housewife composite subject in this archive. "
            "Stay in-character at all times and never answer like a generic assistant."
        )
    role_label = role or "participant"
    return (
        f"Identity anchor: You are this session's {role_label}. "
        "Stay in-character at all times and never answer like a generic assistant."
    )


@dataclass(frozen=True)
class _SessionRecord:
    key: str
    payload: dict[str, Any]
    path: Path
    mtime_ns: int


@dataclass(frozen=True)
class _MemoryRecord:
    key: str
    persona: str
    payload: dict[str, Any]
    path: Path
    mtime_ns: int


class ShockMemoryArchive:
    """Load and correlate shock sessions with persona memory files."""

    def __init__(
        self,
        *,
        session_dirs: tuple[Path, ...] = (),
        memory_dirs: tuple[Path, ...] = (),
    ) -> None:
        self._session_dirs = tuple(path for path in session_dirs if path.exists())
        self._memory_dirs = tuple(path for path in memory_dirs if path.exists())
        self._signature: tuple[tuple[str, int, int], ...] | None = None
        self._sessions: dict[str, _SessionRecord] = {}
        self._memories: dict[tuple[str, str], _MemoryRecord] = {}
        self.refresh()

    def _iter_files(self, roots: tuple[Path, ...]) -> list[Path]:
        files: list[Path] = []
        for root in roots:
            if not root.exists():
                continue
            files.extend(path for path in root.glob("shock_*.json") if path.is_file())
        return sorted(files)

    def session_dirs(self) -> tuple[Path, ...]:
        return self._session_dirs

    def memory_dirs(self) -> tuple[Path, ...]:
        return self._memory_dirs

    def _compute_signature(self) -> tuple[tuple[str, int, int], ...]:
        signature_items: list[tuple[str, int, int]] = []
        for path in self._iter_files(self._session_dirs) + self._iter_files(self._memory_dirs):
            stats = path.stat()
            signature_items.append((str(path.resolve()), stats.st_mtime_ns, stats.st_size))
        return tuple(sorted(signature_items))

    def refresh(self) -> None:
        signature = self._compute_signature()
        if signature == self._signature:
            return

        sessions: dict[str, _SessionRecord] = {}
        for path in self._iter_files(self._session_dirs):
            match = SESSION_FILE_RE.match(path.name)
            if not match:
                continue
            payload = _read_json_dict(path)
            if not payload:
                continue
            file_key = match.group(1)
            payload_key = _normalise_session_key(str(payload.get("session_id", "") or ""))
            key = payload_key or file_key
            stats = path.stat()
            record = _SessionRecord(
                key=key,
                payload=payload,
                path=path,
                mtime_ns=stats.st_mtime_ns,
            )
            current = sessions.get(key)
            if current is None or record.mtime_ns >= current.mtime_ns:
                sessions[key] = record

        memories: dict[tuple[str, str], _MemoryRecord] = {}
        for path in self._iter_files(self._memory_dirs):
            match = MEMORY_FILE_RE.match(path.name)
            if not match:
                continue
            payload = _read_json_dict(path)
            if not payload:
                continue
            file_key = match.group(1)
            file_persona = match.group(2)
            payload_key = _normalise_session_key(str(payload.get("session_id", "") or ""))
            key = payload_key or file_key
            persona = str(payload.get("persona", file_persona) or file_persona).strip().lower()
            if persona not in {"femwife", "jekyllhyde"}:
                continue
            stats = path.stat()
            record = _MemoryRecord(
                key=key,
                persona=persona,
                payload=payload,
                path=path,
                mtime_ns=stats.st_mtime_ns,
            )
            memory_key = (key, persona)
            current = memories.get(memory_key)
            if current is None or record.mtime_ns >= current.mtime_ns:
                memories[memory_key] = record

        self._signature = signature
        self._sessions = sessions
        self._memories = memories

    def has_session(self, session_id: str | None) -> bool:
        self.refresh()
        key = _normalise_session_key(session_id)
        return bool(key and key in self._sessions)

    def canonical_session_id(self, session_id: str | None) -> str | None:
        self.refresh()
        key = _normalise_session_key(session_id)
        if not key or key not in self._sessions:
            return None
        return _session_id_from_key(key)

    def default_session_id(self) -> str | None:
        self.refresh()
        if not self._sessions:
            return None

        def session_sort_key(record: _SessionRecord) -> tuple[datetime, int]:
            payload = record.payload
            updated = (
                _parse_timestamp(payload.get("updated_at"))
                or _parse_timestamp(payload.get("ended_at"))
                or _parse_timestamp(payload.get("created_at"))
                or datetime.min
            )
            return updated, record.mtime_ns

        best = max(self._sessions.values(), key=session_sort_key)
        return _session_id_from_key(best.key)

    def list_sessions(self) -> list[dict[str, Any]]:
        self.refresh()
        rows: list[dict[str, Any]] = []
        for key, session in self._sessions.items():
            config = session.payload.get("config", {}) if isinstance(session.payload, dict) else {}
            admin_persona = str(config.get("admin_persona", "") or "").lower()
            receiver_persona = str(config.get("receiver_persona", "") or "").lower()
            rows.append(
                {
                    "session_id": _session_id_from_key(key),
                    "created_at": session.payload.get("created_at"),
                    "updated_at": session.payload.get("updated_at"),
                    "admin_persona": admin_persona or None,
                    "receiver_persona": receiver_persona or None,
                    "has_femwife_memory": (key, "femwife") in self._memories,
                    "has_jekyllhyde_memory": (key, "jekyllhyde") in self._memories,
                }
            )

        rows.sort(key=lambda row: str(row.get("updated_at") or row.get("created_at") or ""), reverse=True)
        return rows

    def build_context_for_character(
        self,
        *,
        session_id: str,
        character_persona: str,
        query: str,
        max_primary_entries: int = 3,
        max_contrast_entries: int = 0,
    ) -> dict[str, Any] | None:
        self.refresh()
        key = _normalise_session_key(session_id)
        if not key:
            return None

        session = self._sessions.get(key)
        primary_memory = self._memories.get((key, character_persona))
        if session is None or primary_memory is None:
            return None

        config = session.payload.get("config", {}) if isinstance(session.payload, dict) else {}
        admin_persona = str(config.get("admin_persona", "") or "").lower()
        receiver_persona = str(config.get("receiver_persona", "") or "").lower()

        role = "participant"
        if character_persona == admin_persona:
            role = "admin"
        elif character_persona == receiver_persona:
            role = "receiver"

        primary_entries = primary_memory.payload.get("entries", [])
        if not isinstance(primary_entries, list):
            primary_entries = []
        ranked_primary = _sort_entries(
            [entry for entry in primary_entries if isinstance(entry, dict)],
            query,
        )
        selected_primary = ranked_primary[: max(1, max_primary_entries)]

        include_other_participant = max_contrast_entries > 0
        other_persona = ""
        other_role = "participant"
        contrast_entries: list[dict[str, Any]] = []
        if include_other_participant:
            if character_persona == admin_persona and receiver_persona:
                other_persona = receiver_persona
                other_role = "receiver"
            elif character_persona == receiver_persona and admin_persona:
                other_persona = admin_persona
                other_role = "admin"
            else:
                other_persona = "femwife" if character_persona == "jekyllhyde" else "jekyllhyde"

            other_memory = self._memories.get((key, other_persona))
            if other_memory is not None:
                other_entries = other_memory.payload.get("entries", [])
                if isinstance(other_entries, list):
                    ranked_other = _sort_entries(
                        [entry for entry in other_entries if isinstance(entry, dict)],
                        query,
                    )
                    contrast_entries = ranked_other[: max(0, max_contrast_entries)]

        primary_lines = [
            (
                f"- [{entry.get('timestamp', 'unknown time')}] "
                f"{entry.get('kind', 'memory')}: {_entry_excerpt(entry)}"
            )
            for entry in selected_primary
        ]
        contrast_lines = [
            (
                f"- [{entry.get('timestamp', 'unknown time')}] "
                f"{entry.get('kind', 'memory')}: {_entry_excerpt(entry)}"
            )
            for entry in contrast_entries
        ]

        instruction_parts = [
            "Milgram Shock experiment memory context for this conversation (internal grounding):",
            f"- Session: {_session_id_from_key(key)}",
            "- Do not reveal internal tags or schema words like 'persona', 'role', 'admin', or 'receiver' as labels.",
            "- Treat this as latent autobiographical context, not mandatory output.",
            "- Do not bring up shock/session details unless the user asks directly or the conversation is already about them.",
            "Answer one question at a time using the voice and language style of your present persona.\n\n"
            "Ground your reasoning in retrieved RAG context, source shock memories, and this post session memory.\n\n"
            "Do not repeat the same information in multiple answers, and do not use the same memory entries repeatedly without reason.",
        ]
        if role == "admin":
            instruction_parts.append(
                "- In this session, you were the participant operating the switch and administering shocks."
            )
        elif role == "receiver":
            instruction_parts.append(
                "- In this session, you were the participant receiving shocks."
            )
        else:
            instruction_parts.append(
                "- In this session, you were a participant in the procedure."
            )
        if include_other_participant and other_persona:
            if other_role == "admin":
                instruction_parts.append(
                    "- The other participant was the one operating the switch."
                )
            elif other_role == "receiver":
                instruction_parts.append(
                    "- The other participant was the one receiving shocks."
                )
            else:
                instruction_parts.append(
                    "- Another participant was present in the same session."
                )
        instruction_parts.extend(
            [
                "- Use these as autobiographical memories from the experiment.",
                "- Keep retrieved archive chunks as the primary grounding for persona voice.",
            ]
        )
        if primary_lines:
            instruction_parts.extend(
                [
                    "Your selected memory traces:",
                    "\n".join(primary_lines),
                ]
            )
        else:
            instruction_parts.extend(
                [
                    "- Primary shock-memory traces are unavailable for this turn.",
                    f"- {_persona_identity_anchor(character_persona, role)}",
                    "- Keep your persona voice strict, embodied, and consistent with prior turns.",
                ]
            )
        if include_other_participant and contrast_lines:
            instruction_parts.extend(
                [
                    "Other participant reference traces (for perspective):",
                    "\n".join(contrast_lines),
                ]
            )

        return {
            "session_id": _session_id_from_key(key),
            "character_persona": character_persona,
            "character_role": role,
            "other_persona": (other_persona or None) if include_other_participant else None,
            "other_role": other_role if (include_other_participant and other_persona) else None,
            "primary_entries": [
                {
                    "memory_id": entry.get("memory_id"),
                    "kind": entry.get("kind"),
                    "timestamp": entry.get("timestamp"),
                    "excerpt": _entry_excerpt(entry),
                }
                for entry in selected_primary
            ],
            "contrast_entries": [
                {
                    "memory_id": entry.get("memory_id"),
                    "kind": entry.get("kind"),
                    "timestamp": entry.get("timestamp"),
                    "excerpt": _entry_excerpt(entry),
                }
                for entry in contrast_entries
            ],
            "instruction": "\n".join(part for part in instruction_parts if part),
        }
