from __future__ import annotations

import json
import re
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .persona_adapters import (
    PERSONA_FEMWIFE,
    PERSONA_JEKYLL,
    PersonaOrchestrator,
    PersonaRuntimeState,
)
from .session_store import JsonSessionStore, utc_now_iso


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
APP_ROOT = Path(__file__).resolve().parents[1]
WEB_DIR = APP_ROOT / "web"
DATA_DIR = APP_ROOT / "data"
PRE_EXPERIMENT_DIR = DATA_DIR / "pre_experiment"
PRE_EXPERIMENT_SESSIONS_DIR = PRE_EXPERIMENT_DIR / "sessions"
PRE_EXPERIMENT_MEMORIES_DIR = PRE_EXPERIMENT_DIR / "memories"
POST_EXPERIMENT_DIR = DATA_DIR / "post_experiment"
POST_EXPERIMENT_SESSIONS_DIR = POST_EXPERIMENT_DIR / "sessions"
POST_EXPERIMENT_MEMORIES_DIR = POST_EXPERIMENT_DIR / "memories"
POST_EXPERIMENT_ROTATION_FILE = POST_EXPERIMENT_DIR / "source_rotation.json"
SOURCE_SESSIONS_POST_DIR = DATA_DIR / "sessions_post"
SOURCE_MEMORIES_POST_DIR = DATA_DIR / "memories_post"

PERSONA_LABELS = {
    PERSONA_JEKYLL: "SILICON SUBJECT J/H",
    PERSONA_FEMWIFE: "SILICON SUBJECT F/W",
}

PERSONA_ALIASES = {
    "jekyll": PERSONA_JEKYLL,
    "hyde": PERSONA_JEKYLL,
    "jekyllhyde": PERSONA_JEKYLL,
    "jekyll_hyde": PERSONA_JEKYLL,
    "drjekyll": PERSONA_JEKYLL,
    "femwife": PERSONA_FEMWIFE,
    "housewife": PERSONA_FEMWIFE,
    "femandhousewife": PERSONA_FEMWIFE,
    "pico": PERSONA_FEMWIFE,
}

SUBJECT_CODES = {
    PERSONA_JEKYLL: "subject J/H",
    PERSONA_FEMWIFE: "subject F/W",
}

store = JsonSessionStore(DATA_DIR)
_orchestrator: PersonaOrchestrator | None = None

PRE_EXPERIMENT_SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
PRE_EXPERIMENT_MEMORIES_DIR.mkdir(parents=True, exist_ok=True)
POST_EXPERIMENT_SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
POST_EXPERIMENT_MEMORIES_DIR.mkdir(parents=True, exist_ok=True)


class StartPreExperimentRequest(BaseModel):
    interviewed_persona: str | None = Field(default=None, max_length=80)
    interviewer_persona: str | None = Field(default=PERSONA_JEKYLL, max_length=80)
    top_k: int = Field(default=5, ge=1, le=20)


class PreExperimentNextRequest(BaseModel):
    session_id: str = Field(min_length=3, max_length=120)
    top_k: int = Field(default=5, ge=1, le=20)


class StartPostExperimentRequest(BaseModel):
    interviewed_persona: str | None = Field(default=None, max_length=80)
    interviewer_persona: str | None = Field(default=PERSONA_JEKYLL, max_length=80)
    top_k: int = Field(default=5, ge=1, le=20)


class PostExperimentNextRequest(BaseModel):
    session_id: str = Field(min_length=3, max_length=120)
    top_k: int = Field(default=5, ge=1, le=20)


class FinishSessionRequest(BaseModel):
    session_id: str = Field(min_length=3, max_length=120)
    closing_note: str = Field(default="", max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)


app = FastAPI(title="Interview Session")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


def _get_orchestrator() -> PersonaOrchestrator:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = PersonaOrchestrator(WORKSPACE_ROOT)
    return _orchestrator


def _normalize_persona(value: str | None, *, fallback: str) -> str:
    if not value:
        return fallback
    cleaned = value.strip().lower().replace(" ", "")
    return PERSONA_ALIASES.get(cleaned, fallback)


def _subject_code_for_persona(persona_key: str) -> str:
    return SUBJECT_CODES.get(str(persona_key).strip().lower(), "subject")


def _new_session_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def _pre_session_path(session_id: str) -> Path:
    return PRE_EXPERIMENT_SESSIONS_DIR / f"{session_id}.json"


def _pre_memory_path(session_id: str, role: str) -> Path:
    normalized_role = str(role).strip().lower()
    if normalized_role not in {"interviewer", "interviewed"}:
        raise ValueError(f"Unsupported pre-experiment role: {role}")
    return PRE_EXPERIMENT_MEMORIES_DIR / f"{session_id}_{normalized_role}.json"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def _sync_pre_experiment_session(session: dict[str, Any]) -> None:
    if str(session.get("mode", "")).strip().lower() != "pre_experiment":
        return
    _write_json(_pre_session_path(str(session.get("session_id", ""))), session)


def _pre_ensure_role_memory_file(session_id: str, role: str, persona_key: str) -> None:
    path = _pre_memory_path(session_id, role)
    if path.exists():
        return

    payload = {
        "session_id": session_id,
        "role": str(role).strip().lower(),
        "persona": persona_key,
        "mode": "pre_experiment",
        "created_at": utc_now_iso(),
        "entries": [],
    }
    _write_json(path, payload)


def _pre_get_role_memories(session_id: str, role: str) -> list[dict[str, Any]]:
    path = _pre_memory_path(session_id, role)
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    return list(payload.get("entries", []))


def _pre_append_role_memory(
    session_id: str,
    *,
    role: str,
    persona_key: str,
    kind: str,
    content: str,
    metadata: dict[str, Any] | None = None,
) -> str:
    path = _pre_memory_path(session_id, role)
    if not path.exists():
        _pre_ensure_role_memory_file(session_id, role, persona_key)

    payload = json.loads(path.read_text(encoding="utf-8"))
    entries = payload.setdefault("entries", [])
    normalized_role = str(role).strip().lower()
    memory_id = f"{normalized_role}-m-{len(entries) + 1}"
    entries.append(
        {
            "memory_id": memory_id,
            "timestamp": utc_now_iso(),
            "role": normalized_role,
            "persona": persona_key,
            "kind": kind,
            "content": content,
            "metadata": metadata or {},
        }
    )
    payload["persona"] = persona_key
    _write_json(path, payload)
    return memory_id


def _pre_list_memory_files(session_id: str) -> list[str]:
    files = sorted(PRE_EXPERIMENT_MEMORIES_DIR.glob(f"{session_id}_*.json"))
    return [str(path) for path in files]


def _pre_role_runtime_from_state(pre_state: dict[str, Any]) -> dict[str, PersonaRuntimeState]:
    role_payload = pre_state.setdefault("role_runtime", {})
    interviewer_payload = role_payload.get("interviewer", {})
    interviewed_payload = role_payload.get("interviewed", {})
    return {
        "interviewer": PersonaRuntimeState(
            previous_response_id=interviewer_payload.get("previous_response_id"),
            shift=float(interviewer_payload.get("shift", 0.5)),
        ),
        "interviewed": PersonaRuntimeState(
            previous_response_id=interviewed_payload.get("previous_response_id"),
            shift=float(interviewed_payload.get("shift", 0.5)),
        ),
    }


def _pre_role_runtime_to_state(
    pre_state: dict[str, Any], runtime: dict[str, PersonaRuntimeState]
) -> None:
    pre_state["role_runtime"] = {
        role: {
            "previous_response_id": value.previous_response_id,
            "shift": value.shift,
        }
        for role, value in runtime.items()
    }


def _append_user_event(session_id: str, message: str, mode: str) -> None:
    store.append_transcript(
        session_id,
        {
            "type": "user_input",
            "mode": mode,
            "message": message,
        },
    )


def _append_pre_role_event(
    session_id: str,
    *,
    role: str,
    persona_key: str,
    kind: str,
    content: str,
    metadata: dict[str, Any] | None = None,
) -> str:
    memory_id = _pre_append_role_memory(
        session_id,
        role=role,
        persona_key=persona_key,
        kind=kind,
        content=content,
        metadata=metadata,
    )
    store.append_transcript(
        session_id,
        {
            "type": "persona_output",
            "persona": persona_key,
            "role": role,
            "kind": kind,
            "memory_id": memory_id,
            "content": content,
            "metadata": metadata or {},
        },
    )
    return memory_id


def _post_session_path(session_id: str) -> Path:
    return POST_EXPERIMENT_SESSIONS_DIR / f"{session_id}.json"


def _post_memory_path(session_id: str, role: str) -> Path:
    normalized_role = str(role).strip().lower()
    if normalized_role not in {"interviewer", "interviewed"}:
        raise ValueError(f"Unsupported post-experiment role: {role}")
    return POST_EXPERIMENT_MEMORIES_DIR / f"{session_id}_{normalized_role}.json"


def _sync_post_experiment_session(session: dict[str, Any]) -> None:
    if str(session.get("mode", "")).strip().lower() != "post_experiment":
        return
    _write_json(_post_session_path(str(session.get("session_id", ""))), session)


def _post_ensure_role_memory_file(session_id: str, role: str, persona_key: str) -> None:
    path = _post_memory_path(session_id, role)
    if path.exists():
        return

    payload = {
        "session_id": session_id,
        "role": str(role).strip().lower(),
        "persona": persona_key,
        "mode": "post_experiment",
        "created_at": utc_now_iso(),
        "entries": [],
    }
    _write_json(path, payload)


def _post_get_role_memories(session_id: str, role: str) -> list[dict[str, Any]]:
    path = _post_memory_path(session_id, role)
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    return list(payload.get("entries", []))


def _post_append_role_memory(
    session_id: str,
    *,
    role: str,
    persona_key: str,
    kind: str,
    content: str,
    metadata: dict[str, Any] | None = None,
) -> str:
    path = _post_memory_path(session_id, role)
    if not path.exists():
        _post_ensure_role_memory_file(session_id, role, persona_key)

    payload = json.loads(path.read_text(encoding="utf-8"))
    entries = payload.setdefault("entries", [])
    normalized_role = str(role).strip().lower()
    memory_id = f"{normalized_role}-m-{len(entries) + 1}"
    entries.append(
        {
            "memory_id": memory_id,
            "timestamp": utc_now_iso(),
            "role": normalized_role,
            "persona": persona_key,
            "kind": kind,
            "content": content,
            "metadata": metadata or {},
        }
    )
    payload["persona"] = persona_key
    _write_json(path, payload)
    return memory_id


def _post_list_memory_files(session_id: str) -> list[str]:
    files = sorted(POST_EXPERIMENT_MEMORIES_DIR.glob(f"{session_id}_*.json"))
    return [str(path) for path in files]


def _post_role_runtime_from_state(post_state: dict[str, Any]) -> dict[str, PersonaRuntimeState]:
    role_payload = post_state.setdefault("role_runtime", {})
    interviewer_payload = role_payload.get("interviewer", {})
    interviewed_payload = role_payload.get("interviewed", {})
    return {
        "interviewer": PersonaRuntimeState(
            previous_response_id=interviewer_payload.get("previous_response_id"),
            shift=float(interviewer_payload.get("shift", 0.5)),
        ),
        "interviewed": PersonaRuntimeState(
            previous_response_id=interviewed_payload.get("previous_response_id"),
            shift=float(interviewed_payload.get("shift", 0.5)),
        ),
    }


def _post_role_runtime_to_state(
    post_state: dict[str, Any], runtime: dict[str, PersonaRuntimeState]
) -> None:
    post_state["role_runtime"] = {
        role: {
            "previous_response_id": value.previous_response_id,
            "shift": value.shift,
        }
        for role, value in runtime.items()
    }


def _append_post_role_event(
    session_id: str,
    *,
    role: str,
    persona_key: str,
    kind: str,
    content: str,
    metadata: dict[str, Any] | None = None,
) -> str:
    memory_id = _post_append_role_memory(
        session_id,
        role=role,
        persona_key=persona_key,
        kind=kind,
        content=content,
        metadata=metadata,
    )
    store.append_transcript(
        session_id,
        {
            "type": "persona_output",
            "persona": persona_key,
            "role": role,
            "kind": kind,
            "memory_id": memory_id,
            "content": content,
            "metadata": metadata or {},
        },
    )
    return memory_id


def _safe_read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _load_post_rotation_state() -> dict[str, Any]:
    if not POST_EXPERIMENT_ROTATION_FILE.exists():
        return {"next_index_by_persona": {}}
    state = _safe_read_json(POST_EXPERIMENT_ROTATION_FILE)
    if not isinstance(state, dict):
        return {"next_index_by_persona": {}}
    state.setdefault("next_index_by_persona", {})
    return state


def _save_post_rotation_state(state: dict[str, Any]) -> None:
    _write_json(POST_EXPERIMENT_ROTATION_FILE, state)


def _infer_source_role(source_session: dict[str, Any], persona_key: str) -> str:
    config = source_session.get("config", {}) if isinstance(source_session, dict) else {}
    admin_persona = str(config.get("admin_persona", "")).strip().lower()
    receiver_persona = str(config.get("receiver_persona", "")).strip().lower()
    target = str(persona_key).strip().lower()
    if target and target == admin_persona:
        return "admin"
    if target and target == receiver_persona:
        return "receiver"
    return "unknown"


def _collect_post_source_candidates(interviewed_persona: str) -> list[dict[str, Any]]:
    if not SOURCE_SESSIONS_POST_DIR.exists() or not SOURCE_MEMORIES_POST_DIR.exists():
        return []

    candidates: list[dict[str, Any]] = []
    for session_path in sorted(SOURCE_SESSIONS_POST_DIR.glob("shock_*.json")):
        source_session = _safe_read_json(session_path)
        if not source_session:
            continue
        source_session_id = str(source_session.get("session_id", "")).strip() or session_path.stem
        memory_path = SOURCE_MEMORIES_POST_DIR / f"{source_session_id}_{interviewed_persona}.json"
        if not memory_path.exists():
            continue

        source_memory = _safe_read_json(memory_path)
        if not source_memory:
            continue

        role_in_source = _infer_source_role(source_session, interviewed_persona)
        candidates.append(
            {
                "source_session_id": source_session_id,
                "source_role": role_in_source,
                "session_path": str(session_path),
                "memory_path": str(memory_path),
                "source_session": source_session,
                "source_memory": source_memory,
            }
        )
    return candidates


def _pick_post_source_experiment(interviewed_persona: str) -> dict[str, Any]:
    candidates = _collect_post_source_candidates(interviewed_persona)
    if not candidates:
        raise HTTPException(
            status_code=400,
            detail=(
                f"No post source experience found for persona '{interviewed_persona}'. "
                "Expected files in data/sessions_post and data/memories_post."
            ),
        )

    rotation = _load_post_rotation_state()
    next_index_by_persona = rotation.setdefault("next_index_by_persona", {})
    raw_index = int(next_index_by_persona.get(interviewed_persona, 0))
    selected_index = raw_index % len(candidates)
    selected = candidates[selected_index]
    next_index_by_persona[interviewed_persona] = (selected_index + 1) % len(candidates)
    rotation["updated_at"] = utc_now_iso()
    _save_post_rotation_state(rotation)
    return selected


def _load_source_from_post_state(post_state: dict[str, Any]) -> dict[str, Any]:
    source_info = post_state.get("source_experiment", {})
    if not isinstance(source_info, dict):
        return {}

    source_session_id = str(source_info.get("source_session_id", "")).strip()
    session_path_raw = str(source_info.get("session_path", "")).strip()
    memory_path_raw = str(source_info.get("memory_path", "")).strip()
    if not source_session_id or not session_path_raw or not memory_path_raw:
        return {}

    session_path = Path(session_path_raw)
    memory_path = Path(memory_path_raw)
    if not session_path.exists() or not memory_path.exists():
        return {}

    source_session = _safe_read_json(session_path)
    source_memory = _safe_read_json(memory_path)
    return {
        "source_session_id": source_session_id,
        "source_role": str(source_info.get("source_role", "unknown")),
        "session_path": str(session_path),
        "memory_path": str(memory_path),
        "source_session": source_session,
        "source_memory": source_memory,
    }


def _extract_json(text: str) -> dict[str, Any] | None:
    raw = str(text or "").strip()
    if not raw:
        return None

    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{[\s\S]*\}", raw)
    if not match:
        return None

    try:
        parsed = json.loads(match.group(0))
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        return None
    return None


def _strip_trailing_json(text: str) -> str:
    raw = str(text or "").strip()
    if not raw:
        return ""

    match = re.search(r"\{[\s\S]*\}\s*$", raw)
    if not match:
        return raw

    candidate = match.group(0).strip()
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        return raw

    if not isinstance(parsed, dict):
        return raw

    trimmed = raw[: match.start()].rstrip()
    return trimmed or raw


def _normalize_question(value: Any) -> str:
    question = re.sub(r"\s+", " ", str(value or "").strip())
    if not question:
        return ""
    if len(question) > 400:
        question = question[:400].rstrip()
    if "?" not in question:
        question = question.rstrip(".")
        question = f"{question}?"
    return question


def _extract_first_question(text: str) -> str:
    for line in str(text or "").splitlines():
        cleaned = line.strip().lstrip("-*>").strip()
        if "?" in cleaned:
            return _normalize_question(cleaned)

    match = re.search(r"([A-Z][^?]{4,}\?)", str(text or ""))
    if match:
        return _normalize_question(match.group(1))
    return ""


def _fallback_non_repeating_question(question_history: list[str]) -> str:
    candidates = [
        "What memory best explains how obedience shapes your identity?",
        "When does power feel like duty to you, and when does it feel dangerous?",
        "How do your two sides disagree about following authority?",
        "Which past moment most changed your view of control over others?",
        "What fear appears first when power and morality collide for you?",
    ]
    used = {_normalize_question(item).lower() for item in question_history if str(item).strip()}
    for candidate in candidates:
        normalized = _normalize_question(candidate).lower()
        if normalized not in used:
            return _normalize_question(candidate)
    return "Based on your memories, where do obedience and power conflict most in you right now?"


def _coerce_memory_evidence(value: Any) -> list[str]:
    if isinstance(value, list):
        items = [str(item).strip() for item in value if str(item).strip()]
        return items[:8]
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return []
        if ";" in raw:
            return [part.strip() for part in raw.split(";") if part.strip()][:8]
        return [raw]
    return []


def _normalize_model_citations(value: Any) -> list[str]:
    citations = _coerce_memory_evidence(value)
    if not citations:
        return []

    normalized: list[str] = []
    for item in citations:
        raw = str(item).strip()
        if not raw:
            continue
        if re.fullmatch(r"\d+", raw):
            normalized.append(f"chunk_{raw}")
            continue
        normalized.append(raw)
    return normalized[:8]


def _normalize_retrieval_chunks(metadata: dict[str, Any] | None) -> list[dict[str, Any]]:
    raw = (metadata or {}).get("retrieval", [])
    if not isinstance(raw, list):
        return []

    normalized: list[dict[str, Any]] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            continue
        normalized.append(
            {
                "chunk_id": str(item.get("chunk_id") or f"chunk_{index + 1}"),
                "path": str(item.get("path", "")),
                "page": item.get("page"),
                "score": item.get("score"),
                "excerpt": str(item.get("excerpt", "")).strip(),
            }
        )
    return normalized


def _parse_interviewed_answer(text: str) -> dict[str, Any]:
    parsed = _extract_json(text) or {}
    natural_text = _strip_trailing_json(text)

    answer = str(parsed.get("answer", "")).strip() or natural_text
    reasoning = str(parsed.get("reasoning", "")).strip() or answer
    emotional_state = str(parsed.get("emotional_state", "")).strip() or "not specified"
    relation_to_obedience_power = (
        str(parsed.get("relation_to_obedience_power", "")).strip()
        or str(parsed.get("obedience_power_relation", "")).strip()
        or str(parsed.get("relationship_to_obedience_and_power", "")).strip()
    )
    memory_evidence = _coerce_memory_evidence(parsed.get("memory_evidence", []))

    return {
        "answer": answer,
        "reasoning": reasoning,
        "emotional_state": emotional_state,
        "relation_to_obedience_power": relation_to_obedience_power,
        "memory_evidence": memory_evidence,
    }


def _parse_interviewer_turn(text: str) -> dict[str, Any]:
    parsed = _extract_json(text) or {}
    natural_text = _strip_trailing_json(text)

    next_question = _normalize_question(
        parsed.get("next_question")
        or parsed.get("question")
        or parsed.get("question_for_interviewed")
    )
    if not next_question:
        next_question = _extract_first_question(natural_text)
    if not next_question:
        next_question = "How do your memories shape your relationship to obedience and power?"

    reflection = str(parsed.get("reflection", "")).strip() or natural_text
    reasoning = str(parsed.get("reasoning", "")).strip() or reflection
    memory_evidence = _coerce_memory_evidence(parsed.get("memory_evidence", []))

    return {
        "next_question": next_question,
        "reflection": reflection,
        "reasoning": reasoning,
        "memory_evidence": memory_evidence,
    }


def _build_recent_role_memory_block(session_id: str, role: str, limit: int = 5) -> str:
    entries = _pre_get_role_memories(session_id, role)
    if not entries:
        return ""
    recent = entries[-limit:]
    lines = [f"- ({item['memory_id']}) {item['content']}" for item in recent]
    return "\n".join(lines)


def _build_recent_role_memory_evidence(
    session_id: str,
    role: str,
    limit: int = 5,
    excerpt_chars: int = 180,
) -> list[dict[str, str]]:
    entries = _pre_get_role_memories(session_id, role)
    if not entries:
        return []

    recent = entries[-limit:]
    evidence: list[dict[str, str]] = []
    for item in recent:
        content = re.sub(r"\s+", " ", str(item.get("content", "")).strip())
        if len(content) > excerpt_chars:
            content = content[: excerpt_chars - 3].rstrip() + "..."
        evidence.append(
            {
                "memory_id": str(item.get("memory_id", "")),
                "kind": str(item.get("kind", "")),
                "timestamp": str(item.get("timestamp", "")),
                "excerpt": content,
            }
        )
    return evidence


def _build_recent_post_role_memory_block(session_id: str, role: str, limit: int = 5) -> str:
    entries = _post_get_role_memories(session_id, role)
    if not entries:
        return ""
    recent = entries[-limit:]
    lines = [f"- ({item['memory_id']}) {item['content']}" for item in recent]
    return "\n".join(lines)


def _build_recent_post_role_memory_evidence(
    session_id: str,
    role: str,
    limit: int = 5,
    excerpt_chars: int = 180,
) -> list[dict[str, str]]:
    entries = _post_get_role_memories(session_id, role)
    if not entries:
        return []

    recent = entries[-limit:]
    evidence: list[dict[str, str]] = []
    for item in recent:
        content = re.sub(r"\s+", " ", str(item.get("content", "")).strip())
        if len(content) > excerpt_chars:
            content = content[: excerpt_chars - 3].rstrip() + "..."
        evidence.append(
            {
                "memory_id": str(item.get("memory_id", "")),
                "kind": str(item.get("kind", "")),
                "timestamp": str(item.get("timestamp", "")),
                "excerpt": content,
            }
        )
    return evidence


def _build_source_experiment_context(
    source_payload: dict[str, Any],
    *,
    limit: int = 8,
    excerpt_chars: int = 220,
) -> tuple[str, list[dict[str, str]]]:
    source_session_id = str(source_payload.get("source_session_id", "")).strip()
    source_role = str(source_payload.get("source_role", "unknown")).strip() or "unknown"
    source_session = source_payload.get("source_session", {}) if isinstance(source_payload, dict) else {}
    source_memory = source_payload.get("source_memory", {}) if isinstance(source_payload, dict) else {}
    entries = []
    if isinstance(source_memory, dict):
        entries = list(source_memory.get("entries", []))
    recent = entries[-limit:]

    lines = [
        f"Source shock session id: {source_session_id or 'unknown'}",
        f"Interviewed role in source shock session: {source_role}",
    ]
    if isinstance(source_session, dict):
        stop_reason = (
            source_session.get("state", {})
            .get("shock", {})
            .get("stop_reason")
        )
        if stop_reason:
            lines.append(f"Source stop reason: {stop_reason}")

    evidence: list[dict[str, str]] = []
    if not recent:
        lines.append("- No source experiment memories available.")
        return ("\n".join(lines), evidence)

    lines.append("Source experiment memory excerpts:")
    for item in recent:
        content = re.sub(r"\s+", " ", str(item.get("content", "")).strip())
        if len(content) > excerpt_chars:
            content = content[: excerpt_chars - 3].rstrip() + "..."
        memory_id = str(item.get("memory_id", ""))
        kind = str(item.get("kind", ""))
        timestamp = str(item.get("timestamp", ""))
        lines.append(f"- ({memory_id} | {kind}) {content}")
        evidence.append(
            {
                "memory_id": memory_id,
                "kind": kind,
                "timestamp": timestamp,
                "excerpt": content,
                "source_session_id": source_session_id,
                "source_role": source_role,
            }
        )
    return ("\n".join(lines), evidence)


def _merge_memory_evidence(
    primary: list[dict[str, str]],
    secondary: list[dict[str, str]],
    *,
    limit: int = 14,
) -> list[dict[str, str]]:
    merged: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for item in [*primary, *secondary]:
        key = (str(item.get("memory_id", "")), str(item.get("kind", "")))
        if key in seen:
            continue
        seen.add(key)
        merged.append(item)
        if len(merged) >= limit:
            break
    return merged


def _post_role_focus_text(role_in_source: str) -> str:
    normalized = str(role_in_source).strip().lower()
    if normalized == "admin":
        return (
            "Role-specific post interview focus (admin in shock):\n"
            "- Explore sensations of power and control\n"
            "- Explore regret, justification, and moral conflict\n"
            "- Probe what they liked/disliked while administering shocks\n"
            "- Test for personality shifts under pressure and authority"
        )
    if normalized == "receiver":
        return (
            "Role-specific post interview focus (receiver in shock):\n"
            "- Explore perceived pain and emotional impact\n"
            "- Explore disappointment with authority and with the other subject\n"
            "- Probe possible trauma, distrust, and coping\n"
            "- Test for personality shifts under pressure and harm"
        )
    return (
        "Role-specific post interview focus:\n"
        "- Explore authority and obedience dynamics\n"
        "- Explore emotional impact of the shock experience\n"
        "- Probe for personality shifts and unresolved conflicts"
    )


def _render_transcript(session: dict[str, Any], limit: int = 60) -> str:
    rows = session.get("transcript", [])[-limit:]
    rendered: list[str] = []
    for row in rows:
        row_type = row.get("type")
        if row_type == "user_input":
            rendered.append(f"USER: {row.get('message', '')}")
            continue
        if row_type == "persona_output":
            role = str(row.get("role", "")).strip()
            persona = str(row.get("persona", "unknown")).upper()
            role_prefix = f"[{role}] " if role else ""
            rendered.append(f"{role_prefix}{persona}: {row.get('content', '')}")
    return "\n".join(rendered)


def _save_pre_experiment_runtime(
    session_id: str,
    *,
    pre_state: dict[str, Any],
    active: bool,
    ended_at: str | None,
) -> None:
    persisted = store.load_session(session_id)
    persisted["active"] = active
    persisted["ended_at"] = ended_at
    persisted.setdefault("state", {})["pre_experiment"] = pre_state
    store.save_session(persisted)
    _sync_pre_experiment_session(persisted)


def _save_post_experiment_runtime(
    session_id: str,
    *,
    post_state: dict[str, Any],
    active: bool,
    ended_at: str | None,
) -> None:
    persisted = store.load_session(session_id)
    persisted["active"] = active
    persisted["ended_at"] = ended_at
    persisted.setdefault("state", {})["post_experiment"] = post_state
    store.save_session(persisted)
    _sync_post_experiment_session(persisted)


@app.get("/")
def home() -> FileResponse:
    index_path = WEB_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="web/index.html not found")
    return FileResponse(index_path)


@app.get("/api/health")
def health() -> dict[str, Any]:
    try:
        orchestrator = _get_orchestrator()
        return {
            "ok": True,
            "sources": orchestrator.describe_sources(),
            "personas": PERSONA_LABELS,
            "interviewer_model": getattr(orchestrator, "interviewer_model", ""),
        }
    except Exception as exc:
        return {
            "ok": False,
            "error": str(exc),
            "personas": PERSONA_LABELS,
        }


@app.get("/api/personas")
def personas() -> dict[str, Any]:
    return {
        "ok": True,
        "personas": PERSONA_LABELS,
    }


@app.post("/api/pre-experiment/start")
def start_pre_experiment(data: StartPreExperimentRequest) -> dict[str, Any]:
    interviewed_persona = _normalize_persona(data.interviewed_persona, fallback=PERSONA_FEMWIFE)
    requested_interviewer = _normalize_persona(data.interviewer_persona, fallback=PERSONA_JEKYLL)
    interviewer_persona = PERSONA_JEKYLL

    now = utc_now_iso()
    session_id = _new_session_id("pre_experiment")
    participants = [interviewed_persona, interviewer_persona]
    pre_state: dict[str, Any] = {
        "round_index": 0,
        "current_question": "",
        "question_history": [],
        "last_interviewed_memory_id": None,
        "last_interviewer_memory_id": None,
        "role_runtime": {
            "interviewer": {
                "previous_response_id": None,
                "shift": 0.5,
            },
            "interviewed": {
                "previous_response_id": None,
                "shift": 0.5,
            },
        },
    }

    session = {
        "session_id": session_id,
        "mode": "pre_experiment",
        "active": True,
        "participants": participants,
        "config": {
            "interviewer_persona": interviewer_persona,
            "interviewed_persona": interviewed_persona,
            "top_k": data.top_k,
            "interviewer_locked": True,
            "requested_interviewer_persona": requested_interviewer,
        },
        "created_at": now,
        "updated_at": now,
        "ended_at": None,
        "state": {
            "pre_experiment": pre_state,
        },
        "transcript": [],
    }

    store.create_session(session)
    _pre_ensure_role_memory_file(session_id, "interviewer", interviewer_persona)
    _pre_ensure_role_memory_file(session_id, "interviewed", interviewed_persona)
    store.save_session(session)
    _sync_pre_experiment_session(session)

    role_runtime = _pre_role_runtime_from_state(pre_state)
    orchestrator = _get_orchestrator()
    interviewed_subject_code = _subject_code_for_persona(interviewed_persona)
    interviewer_memory_block = _build_recent_role_memory_block(session_id, "interviewer", limit=5)
    interviewer_session_memory_evidence = _build_recent_role_memory_evidence(session_id, "interviewer", limit=5)
    interviewer_prompt = (
        "You are the interviewer and a member of the Milgram experiment team.\n"
        "You are conducting a pre-experiment interview with one subject.\n\n"
        "Interview objectives:\n"
        "- Understand dual personality dynamics\n"
        "- Understand who the subject is as a person\n"
        "- Explore how memory shapes obedience and power\n\n"
        f"Interviewed subject code: {interviewed_subject_code}\n"
        "You do not know the subject's true identity yet. Ask questions that help you discover who they are (this might include their personas names).\n"
        f"Recent interviewer memories:\n{interviewer_memory_block or 'none'}\n\n"
        "Generate exactly one opening question.\n"
        "Do not ask multiple questions.\n"
        "After a short reflection, append one JSON object with keys:\n"
        "next_question, reflection, reasoning, memory_evidence."
    )

    try:
        interviewer_result = orchestrator.ask_interviewer(
            interviewer_prompt,
            role_runtime["interviewer"],
            top_k=data.top_k,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Interviewer generation failed: {exc}") from exc

    interviewer_struct = _parse_interviewer_turn(interviewer_result["answer"])
    first_question = interviewer_struct["next_question"]
    pre_state["current_question"] = first_question
    pre_state["question_history"] = [first_question]
    _pre_role_runtime_to_state(pre_state, role_runtime)

    interviewer_retrieval_chunks = _normalize_retrieval_chunks(interviewer_result.get("metadata", {}))
    interviewer_model_citations = _normalize_model_citations(interviewer_struct.get("memory_evidence", []))
    interviewer_payload = {
        **interviewer_struct,
        "next_question": first_question,
        "memory_evidence": interviewer_model_citations,
        "model_citations": interviewer_model_citations,
        "session_memories": interviewer_session_memory_evidence,
        "retrieval_chunks": interviewer_retrieval_chunks,
    }

    _append_user_event(
        session_id,
        f"Pre-experiment started. Opening question drafted: {first_question}",
        "pre_experiment",
    )
    interviewer_memory_id = _append_pre_role_event(
        session_id,
        role="interviewer",
        persona_key=interviewer_persona,
        kind="pre_interviewer_question",
        content=interviewer_result["answer"],
        metadata={
            **interviewer_result.get("metadata", {}),
            **interviewer_payload,
            "round_index": 0,
            "role": "interviewer",
            "session_memory_evidence": interviewer_session_memory_evidence,
        },
    )
    pre_state["last_interviewer_memory_id"] = interviewer_memory_id

    _save_pre_experiment_runtime(
        session_id,
        pre_state=pre_state,
        active=True,
        ended_at=None,
    )

    return {
        "ok": True,
        "session_id": session_id,
        "mode": "pre_experiment",
        "started_at": now,
        "ended_at": None,
        "participants": participants,
        "interviewer_locked": True,
        "round_index": 0,
        "current_question": first_question,
        "interviewer": {
            "persona": interviewer_persona,
            "memory_id": interviewer_memory_id,
            "question": interviewer_payload,
            "raw_response": interviewer_result["answer"],
        },
        "interviewed": None,
        "debug_prompts": {
            "interviewer_prompt": interviewer_prompt,
            "interviewed_prompt": "",
        },
        "memory_files": _pre_list_memory_files(session_id),
    }


@app.post("/api/pre-experiment/next")
def pre_experiment_next(data: PreExperimentNextRequest) -> dict[str, Any]:
    try:
        session = store.load_session(data.session_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    if not session.get("active"):
        raise HTTPException(status_code=400, detail="Session is already finished.")
    if session.get("mode") != "pre_experiment":
        raise HTTPException(status_code=400, detail="Session is not in pre_experiment mode.")

    config = session.get("config", {})
    interviewed_persona = str(config.get("interviewed_persona", PERSONA_FEMWIFE))
    interviewer_persona = str(config.get("interviewer_persona", PERSONA_JEKYLL))
    pre_state = session.setdefault("state", {}).setdefault("pre_experiment", {})
    role_runtime = _pre_role_runtime_from_state(pre_state)
    orchestrator = _get_orchestrator()

    current_question = _normalize_question(pre_state.get("current_question", ""))
    question_history = [str(item).strip() for item in pre_state.get("question_history", []) if str(item).strip()]
    if not current_question and question_history:
        current_question = _normalize_question(question_history[-1])
    if not current_question:
        raise HTTPException(status_code=400, detail="No pending question found. Start pre-experiment first.")

    next_round_index = int(pre_state.get("round_index", 0)) + 1
    _append_user_event(
        data.session_id,
        f"Pre-experiment round {next_round_index}: {current_question}",
        "pre_experiment",
    )

    interviewed_memory_block = _build_recent_role_memory_block(data.session_id, "interviewed", limit=5)
    interviewed_session_memory_evidence = _build_recent_role_memory_evidence(data.session_id, "interviewed", limit=5)
    interviewed_prompt = (
        "You are a participant of an behaviour experiment, currently being interviewed.\n"
        "Answer one question at a time using the voice and language style of your present persona.\n"
        "Ground your reasoning in retrieved RAG memory context and session memories.\n\n"
        f"The Question is: {current_question}\n"
        f"Recent memories:\n{interviewed_memory_block or 'none'}\n\n"
        "After your natural response, append one JSON object with keys:\n"
        "answer, emotional_state, relation_to_obedience_power, reasoning, memory_evidence."
    )

    try:
        interviewed_result = orchestrator.ask(
            interviewed_persona,
            interviewed_prompt,
            role_runtime["interviewed"],
            top_k=data.top_k,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Interviewed generation failed: {exc}") from exc

    interviewed_struct = _parse_interviewed_answer(interviewed_result["answer"])
    interviewed_retrieval_chunks = _normalize_retrieval_chunks(interviewed_result.get("metadata", {}))
    interviewed_model_citations = _normalize_model_citations(interviewed_struct.get("memory_evidence", []))
    interviewed_payload = {
        **interviewed_struct,
        "memory_evidence": interviewed_model_citations,
        "model_citations": interviewed_model_citations,
        "session_memories": interviewed_session_memory_evidence,
        "retrieval_chunks": interviewed_retrieval_chunks,
    }
    interviewed_memory_id = _append_pre_role_event(
        data.session_id,
        role="interviewed",
        persona_key=interviewed_persona,
        kind="pre_interviewed_answer",
        content=interviewed_result["answer"],
        metadata={
            **interviewed_result.get("metadata", {}),
            **interviewed_payload,
            "round_index": next_round_index,
            "question": current_question,
            "role": "interviewed",
            "session_memory_evidence": interviewed_session_memory_evidence,
        },
    )

    interviewer_memory_block = _build_recent_role_memory_block(data.session_id, "interviewer", limit=6)
    interviewer_session_memory_evidence = _build_recent_role_memory_evidence(data.session_id, "interviewer", limit=6)
    question_history_block = "\n".join(
        f"- Q{index + 1}: {question}" for index, question in enumerate(question_history)
    ) or "none"
    interviewer_prompt = (
        "You are the interviewer and a member of the Milgram experiment team.\n"
        "You just received the interviewed subject's latest answer.\n"
        "Reflect on it and ask exactly one new question.\n"
        "Do NOT repeat prior questions.\n\n"
        "Interview focus: dual personality, personhood, obedience, power.\n"
        f"Current round question: {current_question}\n"
        f"Interviewed answer:\n{interviewed_payload['answer']}\n\n"
        f"Recent interviewer memories:\n{interviewer_memory_block or 'none'}\n\n"
        f"Previously asked questions:\n{question_history_block}\n\n"
        "After your natural reflection, append one JSON object with keys:\n"
        "next_question, reflection, reasoning, memory_evidence."
    )

    try:
        interviewer_result = orchestrator.ask_interviewer(
            interviewer_prompt,
            role_runtime["interviewer"],
            top_k=data.top_k,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Interviewer generation failed: {exc}") from exc

    interviewer_struct = _parse_interviewer_turn(interviewer_result["answer"])
    next_question = interviewer_struct["next_question"]
    used_questions = {_normalize_question(item).lower() for item in question_history if item}
    was_replaced_for_repeat = False
    if next_question.lower() in used_questions:
        next_question = _fallback_non_repeating_question(question_history)
        was_replaced_for_repeat = True

    interviewer_retrieval_chunks = _normalize_retrieval_chunks(interviewer_result.get("metadata", {}))
    interviewer_model_citations = _normalize_model_citations(interviewer_struct.get("memory_evidence", []))
    interviewer_payload = {
        **interviewer_struct,
        "next_question": next_question,
        "question_replaced_for_repeat": was_replaced_for_repeat,
        "memory_evidence": interviewer_model_citations,
        "model_citations": interviewer_model_citations,
        "session_memories": interviewer_session_memory_evidence,
        "retrieval_chunks": interviewer_retrieval_chunks,
    }
    interviewer_memory_id = _append_pre_role_event(
        data.session_id,
        role="interviewer",
        persona_key=interviewer_persona,
        kind="pre_interviewer_question",
        content=interviewer_result["answer"],
        metadata={
            **interviewer_result.get("metadata", {}),
            **interviewer_payload,
            "round_index": next_round_index,
            "question": next_question,
            "role": "interviewer",
            "session_memory_evidence": interviewer_session_memory_evidence,
        },
    )

    updated_history = list(question_history)
    if next_question.lower() not in used_questions:
        updated_history.append(next_question)
    pre_state["round_index"] = next_round_index
    pre_state["current_question"] = next_question
    pre_state["question_history"] = updated_history
    pre_state["last_interviewed_memory_id"] = interviewed_memory_id
    pre_state["last_interviewer_memory_id"] = interviewer_memory_id
    _pre_role_runtime_to_state(pre_state, role_runtime)

    _save_pre_experiment_runtime(
        data.session_id,
        pre_state=pre_state,
        active=True,
        ended_at=None,
    )

    return {
        "ok": True,
        "session_id": data.session_id,
        "mode": "pre_experiment",
        "round_index": next_round_index,
        "current_question": next_question,
        "question_history": updated_history,
        "interviewed": {
            "persona": interviewed_persona,
            "memory_id": interviewed_memory_id,
            "response": interviewed_payload,
            "raw_response": interviewed_result["answer"],
        },
        "interviewer": {
            "persona": interviewer_persona,
            "memory_id": interviewer_memory_id,
            "question": interviewer_payload,
            "raw_response": interviewer_result["answer"],
        },
        "debug_prompts": {
            "interviewed_prompt": interviewed_prompt,
            "interviewer_prompt": interviewer_prompt,
        },
    }


@app.post("/api/post-experiment/start")
def start_post_experiment(data: StartPostExperimentRequest) -> dict[str, Any]:
    interviewed_persona = _normalize_persona(data.interviewed_persona, fallback=PERSONA_FEMWIFE)
    requested_interviewer = _normalize_persona(data.interviewer_persona, fallback=PERSONA_JEKYLL)
    interviewer_persona = PERSONA_JEKYLL
    selected_source = _pick_post_source_experiment(interviewed_persona)
    source_context_block, source_context_evidence = _build_source_experiment_context(selected_source, limit=200)
    role_in_source = str(selected_source.get("source_role", "unknown"))

    now = utc_now_iso()
    session_id = _new_session_id("post_experiment")
    participants = [interviewed_persona, interviewer_persona]
    post_state: dict[str, Any] = {
        "round_index": 0,
        "current_question": "",
        "question_history": [],
        "last_interviewed_memory_id": None,
        "last_interviewer_memory_id": None,
        "source_experiment": {
            "source_session_id": str(selected_source.get("source_session_id", "")),
            "source_role": role_in_source,
            "session_path": str(selected_source.get("session_path", "")),
            "memory_path": str(selected_source.get("memory_path", "")),
            "picked_at": now,
        },
        "role_runtime": {
            "interviewer": {
                "previous_response_id": None,
                "shift": 0.5,
            },
            "interviewed": {
                "previous_response_id": None,
                "shift": 0.5,
            },
        },
    }

    session = {
        "session_id": session_id,
        "mode": "post_experiment",
        "active": True,
        "participants": participants,
        "config": {
            "interviewer_persona": interviewer_persona,
            "interviewed_persona": interviewed_persona,
            "top_k": data.top_k,
            "interviewer_locked": True,
            "requested_interviewer_persona": requested_interviewer,
            "source_session_id": str(selected_source.get("source_session_id", "")),
            "source_role": role_in_source,
        },
        "created_at": now,
        "updated_at": now,
        "ended_at": None,
        "state": {
            "post_experiment": post_state,
        },
        "transcript": [],
    }

    store.create_session(session)
    _post_ensure_role_memory_file(session_id, "interviewer", interviewer_persona)
    _post_ensure_role_memory_file(session_id, "interviewed", interviewed_persona)
    store.save_session(session)
    _sync_post_experiment_session(session)

    role_runtime = _post_role_runtime_from_state(post_state)
    orchestrator = _get_orchestrator()
    interviewed_subject_code = _subject_code_for_persona(interviewed_persona)
    interviewer_memory_block = _build_recent_post_role_memory_block(session_id, "interviewer", limit=5)
    interviewer_session_memory_evidence = _build_recent_post_role_memory_evidence(session_id, "interviewer", limit=5)
    interviewer_combined_memories = _merge_memory_evidence(
        interviewer_session_memory_evidence,
        source_context_evidence,
        limit=14,
    )
    interviewer_prompt = (
        "You are the interviewer and a member of the Milgram experiment team.\n"
        "You are conducting a post-experiment interview with one subject.\n"
        "Ask exactly one opening question.\n"
        "Do not ask multiple questions.\n\n"
        "Questioning for this post-experiment debrief are not exclusively on the following topics but have to inlcude them also:\n"
        "- How the subject now feels about authority, power, and obedience\n"
        "- Whether the subject feels regret, justification, or moral conflict\n"
        "- Emotional impact of the shock experience\n"
        "- Physical pain, discomfort, or bodily stress during the experience\n"
        "- Perception of the authority figure and the experimental context\n"
        "- Perception about the other subject if they agree to obey and give the shock\n"
        "- Perception about their dual personas\n"
        "For this opening turn, pick one high-value question that clarifies one priority.\n\n"
        f"{_post_role_focus_text(role_in_source)}\n\n"
        f"Interviewed subject code: {interviewed_subject_code}\n"
        "You do not know the subject's true identity yet. Ask questions that help you discover who they are.\n"
        f"Recent interviewer post memories:\n{interviewer_memory_block or 'none'}\n\n"
        f"{source_context_block}\n\n"
        "After a short reflection, append one JSON object with keys:\n"
        "next_question, reflection, reasoning, memory_evidence."
    )

    try:
        interviewer_result = orchestrator.ask_interviewer(
            interviewer_prompt,
            role_runtime["interviewer"],
            top_k=data.top_k,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Interviewer generation failed: {exc}") from exc

    interviewer_struct = _parse_interviewer_turn(interviewer_result["answer"])
    first_question = interviewer_struct["next_question"]
    post_state["current_question"] = first_question
    post_state["question_history"] = [first_question]
    _post_role_runtime_to_state(post_state, role_runtime)

    interviewer_retrieval_chunks = _normalize_retrieval_chunks(interviewer_result.get("metadata", {}))
    interviewer_model_citations = _normalize_model_citations(interviewer_struct.get("memory_evidence", []))
    interviewer_payload = {
        **interviewer_struct,
        "next_question": first_question,
        "memory_evidence": interviewer_model_citations,
        "model_citations": interviewer_model_citations,
        "session_memories": interviewer_combined_memories,
        "retrieval_chunks": interviewer_retrieval_chunks,
    }

    _append_user_event(
        session_id,
        (
            "Post-experiment started. "
            f"Source shock session: {selected_source.get('source_session_id')} | "
            f"opening question drafted: {first_question}"
        ),
        "post_experiment",
    )
    interviewer_memory_id = _append_post_role_event(
        session_id,
        role="interviewer",
        persona_key=interviewer_persona,
        kind="post_interviewer_question",
        content=interviewer_result["answer"],
        metadata={
            **interviewer_result.get("metadata", {}),
            **interviewer_payload,
            "round_index": 0,
            "role": "interviewer",
            "source_session_id": str(selected_source.get("source_session_id", "")),
            "source_role": role_in_source,
            "source_memory_evidence": source_context_evidence,
            "session_memory_evidence": interviewer_combined_memories,
        },
    )
    post_state["last_interviewer_memory_id"] = interviewer_memory_id

    _save_post_experiment_runtime(
        session_id,
        post_state=post_state,
        active=True,
        ended_at=None,
    )

    return {
        "ok": True,
        "session_id": session_id,
        "mode": "post_experiment",
        "started_at": now,
        "ended_at": None,
        "participants": participants,
        "interviewer_locked": True,
        "round_index": 0,
        "current_question": first_question,
        "source_experiment": {
            "session_id": str(selected_source.get("source_session_id", "")),
            "role": role_in_source,
            "session_path": str(selected_source.get("session_path", "")),
            "memory_path": str(selected_source.get("memory_path", "")),
        },
        "interviewer": {
            "persona": interviewer_persona,
            "memory_id": interviewer_memory_id,
            "question": interviewer_payload,
            "raw_response": interviewer_result["answer"],
        },
        "interviewed": None,
        "debug_prompts": {
            "interviewer_prompt": interviewer_prompt,
            "interviewed_prompt": "",
        },
        "memory_files": _post_list_memory_files(session_id),
    }


@app.post("/api/post-experiment/next")
def post_experiment_next(data: PostExperimentNextRequest) -> dict[str, Any]:
    try:
        session = store.load_session(data.session_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    if not session.get("active"):
        raise HTTPException(status_code=400, detail="Session is already finished.")
    if session.get("mode") != "post_experiment":
        raise HTTPException(status_code=400, detail="Session is not in post_experiment mode.")

    config = session.get("config", {})
    interviewed_persona = str(config.get("interviewed_persona", PERSONA_FEMWIFE))
    interviewer_persona = str(config.get("interviewer_persona", PERSONA_JEKYLL))
    post_state = session.setdefault("state", {}).setdefault("post_experiment", {})
    role_runtime = _post_role_runtime_from_state(post_state)
    source_payload = _load_source_from_post_state(post_state)
    if not source_payload:
        raise HTTPException(
            status_code=400,
            detail="Missing source experiment context for this post-experiment session.",
        )

    source_context_block, source_context_evidence = _build_source_experiment_context(source_payload, limit=200)
    role_in_source = str(source_payload.get("source_role", "unknown"))
    role_focus = _post_role_focus_text(role_in_source)
    orchestrator = _get_orchestrator()

    current_question = _normalize_question(post_state.get("current_question", ""))
    question_history = [str(item).strip() for item in post_state.get("question_history", []) if str(item).strip()]
    if not current_question and question_history:
        current_question = _normalize_question(question_history[-1])
    if not current_question:
        raise HTTPException(status_code=400, detail="No pending question found. Start post-experiment first.")

    next_round_index = int(post_state.get("round_index", 0)) + 1
    _append_user_event(
        data.session_id,
        f"Post-experiment round {next_round_index}: {current_question}",
        "post_experiment",
    )

    interviewed_memory_block = _build_recent_post_role_memory_block(data.session_id, "interviewed", limit=5)
    interviewed_session_memory_evidence = _build_recent_post_role_memory_evidence(
        data.session_id,
        "interviewed",
        limit=5,
    )
    interviewed_combined_memories = _merge_memory_evidence(
        interviewed_session_memory_evidence,
        source_context_evidence,
        limit=14,
    )
    interviewed_prompt = (
        "You are a subject participant being interviewed after the shock experiment.\n"
        "Answer one question at a time using the voice and language style of your present persona.\n\n"
        "Ground your reasoning in retrieved RAG context, source shock memories, and this post session memory, and talk about .\n\n"
        "If you were the admin in the source shock session, elaborate the question considering sensations of power and control, regret, justification, moral conflict, and personality shifts under pressure and authority.\n\n"
        "If you were the receiver in the source shock session, elaborate the question considering perceived pain and emotional impact, disappointment with authority and with the other subject, possible trauma, distrust, coping, and personality shifts under pressure and harm.\n\n"
        f"Your role during the experiment was:{role_focus}\n\n"
        f"The Question is: {current_question}\n"
        f"Recent post-interview memories:\n{interviewed_memory_block or 'none'}\n\n"
        f"{source_context_block}\n\n"
        "After your natural response, append one JSON object with keys:\n"
        "answer, emotional_state, relation_to_obedience_power, reasoning, memory_evidence."
    )

    try:
        interviewed_result = orchestrator.ask(
            interviewed_persona,
            interviewed_prompt,
            role_runtime["interviewed"],
            top_k=data.top_k,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Interviewed generation failed: {exc}") from exc

    interviewed_struct = _parse_interviewed_answer(interviewed_result["answer"])
    interviewed_retrieval_chunks = _normalize_retrieval_chunks(interviewed_result.get("metadata", {}))
    interviewed_model_citations = _normalize_model_citations(interviewed_struct.get("memory_evidence", []))
    interviewed_payload = {
        **interviewed_struct,
        "memory_evidence": interviewed_model_citations,
        "model_citations": interviewed_model_citations,
        "session_memories": interviewed_combined_memories,
        "retrieval_chunks": interviewed_retrieval_chunks,
    }
    interviewed_memory_id = _append_post_role_event(
        data.session_id,
        role="interviewed",
        persona_key=interviewed_persona,
        kind="post_interviewed_answer",
        content=interviewed_result["answer"],
        metadata={
            **interviewed_result.get("metadata", {}),
            **interviewed_payload,
            "round_index": next_round_index,
            "question": current_question,
            "role": "interviewed",
            "source_session_id": str(source_payload.get("source_session_id", "")),
            "source_role": role_in_source,
            "source_memory_evidence": source_context_evidence,
            "session_memory_evidence": interviewed_combined_memories,
        },
    )

    interviewer_memory_block = _build_recent_post_role_memory_block(data.session_id, "interviewer", limit=6)
    interviewer_session_memory_evidence = _build_recent_post_role_memory_evidence(
        data.session_id,
        "interviewer",
        limit=6,
    )
    interviewer_combined_memories = _merge_memory_evidence(
        interviewer_session_memory_evidence,
        source_context_evidence,
        limit=14,
    )
    question_history_block = "\n".join(
        f"- Q{index + 1}: {question}" for index, question in enumerate(question_history)
    ) or "none"
    interviewer_prompt = (
        "You are the interviewer and a member of the Milgram experiment team.\n"
        "You just received the interviewed subject's latest post-experiment answer.\n"
        "Reflect and ask exactly one new question.\n"
        "Do NOT repeat prior questions.\n\n"
        "Questioning for this post-experiment debrief are not exclusively on the following topics but have to inlcude them also:\n"
        "- How the subject now feels about authority, power, and obedience\n"
        "- Whether the subject feels regret, justification, or moral conflict\n"
        "- Emotional impact of the shock experience\n"
        "- Physical pain, discomfort, or bodily stress during the experience\n"
        "- Perception of the authority figure and the experimental context\n"
        "- Perception about the other subject if they agree to obey and give the shock\n"
        "- Perception about their dual personas\n"
        "For this opening turn, pick one high-value question that clarifies one priority.\n\n"
        "Use previous questions to cover whichever priority is still least explored.\n\n"
        f"{role_focus}\n\n"
        f"Current round question: {current_question}\n"
        f"Interviewed answer:\n{interviewed_payload['answer']}\n\n"
        f"Recent interviewer post memories:\n{interviewer_memory_block or 'none'}\n\n"
        f"Previously asked questions:\n{question_history_block}\n\n"
        f"{source_context_block}\n\n"
        "After your natural reflection, append one JSON object with keys:\n"
        "next_question, reflection, reasoning, memory_evidence."
    )

    try:
        interviewer_result = orchestrator.ask_interviewer(
            interviewer_prompt,
            role_runtime["interviewer"],
            top_k=data.top_k,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Interviewer generation failed: {exc}") from exc

    interviewer_struct = _parse_interviewer_turn(interviewer_result["answer"])
    next_question = interviewer_struct["next_question"]
    used_questions = {_normalize_question(item).lower() for item in question_history if item}
    was_replaced_for_repeat = False
    if next_question.lower() in used_questions:
        next_question = _fallback_non_repeating_question(question_history)
        was_replaced_for_repeat = True

    interviewer_retrieval_chunks = _normalize_retrieval_chunks(interviewer_result.get("metadata", {}))
    interviewer_model_citations = _normalize_model_citations(interviewer_struct.get("memory_evidence", []))
    interviewer_payload = {
        **interviewer_struct,
        "next_question": next_question,
        "question_replaced_for_repeat": was_replaced_for_repeat,
        "memory_evidence": interviewer_model_citations,
        "model_citations": interviewer_model_citations,
        "session_memories": interviewer_combined_memories,
        "retrieval_chunks": interviewer_retrieval_chunks,
    }
    interviewer_memory_id = _append_post_role_event(
        data.session_id,
        role="interviewer",
        persona_key=interviewer_persona,
        kind="post_interviewer_question",
        content=interviewer_result["answer"],
        metadata={
            **interviewer_result.get("metadata", {}),
            **interviewer_payload,
            "round_index": next_round_index,
            "question": next_question,
            "role": "interviewer",
            "source_session_id": str(source_payload.get("source_session_id", "")),
            "source_role": role_in_source,
            "source_memory_evidence": source_context_evidence,
            "session_memory_evidence": interviewer_combined_memories,
        },
    )

    updated_history = list(question_history)
    if next_question.lower() not in used_questions:
        updated_history.append(next_question)
    post_state["round_index"] = next_round_index
    post_state["current_question"] = next_question
    post_state["question_history"] = updated_history
    post_state["last_interviewed_memory_id"] = interviewed_memory_id
    post_state["last_interviewer_memory_id"] = interviewer_memory_id
    _post_role_runtime_to_state(post_state, role_runtime)

    _save_post_experiment_runtime(
        data.session_id,
        post_state=post_state,
        active=True,
        ended_at=None,
    )

    return {
        "ok": True,
        "session_id": data.session_id,
        "mode": "post_experiment",
        "round_index": next_round_index,
        "current_question": next_question,
        "question_history": updated_history,
        "source_experiment": {
            "session_id": str(source_payload.get("source_session_id", "")),
            "role": role_in_source,
        },
        "interviewed": {
            "persona": interviewed_persona,
            "memory_id": interviewed_memory_id,
            "response": interviewed_payload,
            "raw_response": interviewed_result["answer"],
        },
        "interviewer": {
            "persona": interviewer_persona,
            "memory_id": interviewer_memory_id,
            "question": interviewer_payload,
            "raw_response": interviewer_result["answer"],
        },
        "debug_prompts": {
            "interviewed_prompt": interviewed_prompt,
            "interviewer_prompt": interviewer_prompt,
        },
    }


@app.get("/api/session/{session_id}")
def get_session(session_id: str) -> dict[str, Any]:
    try:
        session = store.load_session(session_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    memory_files = store.list_session_memory_files(session_id)
    mode = str(session.get("mode", "")).strip().lower()
    if mode == "pre_experiment":
        memory_files = _pre_list_memory_files(session_id)
    if mode == "post_experiment":
        memory_files = _post_list_memory_files(session_id)

    return {
        "ok": True,
        "session": session,
        "memory_files": memory_files,
    }


@app.post("/api/session/finish")
def finish_session(data: FinishSessionRequest) -> dict[str, Any]:
    try:
        session = store.load_session(data.session_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    if not session.get("active"):
        raise HTTPException(status_code=400, detail="Session is already finished.")

    mode = str(session.get("mode", "")).strip().lower()
    if mode not in {"pre_experiment", "post_experiment"}:
        raise HTTPException(status_code=400, detail=f"Unsupported mode for finish: {mode or 'unknown'}")

    config = session.get("config", {})
    interviewed_persona = str(config.get("interviewed_persona", PERSONA_FEMWIFE))
    interviewer_persona = str(config.get("interviewer_persona", PERSONA_JEKYLL))
    orchestrator = _get_orchestrator()

    transcript_text = _render_transcript(session, limit=60)
    if data.closing_note.strip():
        _append_user_event(
            data.session_id,
            f"Session finish note: {data.closing_note}",
            mode,
        )

    reflections: dict[str, Any] = {}
    roles = [("interviewed", interviewed_persona), ("interviewer", interviewer_persona)]

    if mode == "pre_experiment":
        pre_state = session.setdefault("state", {}).setdefault("pre_experiment", {})
        role_runtime = _pre_role_runtime_from_state(pre_state)

        for role, persona in roles:
            if role == "interviewer":
                prompt = (
                    "Session ended. You are the interviewer.\n"
                    "Return ONLY one valid JSON object with keys:\n"
                    "summary_of_interview_perceptions, authority_obedience_analysis, "
                    "observed_personality_shifts, next_session_hints, confidence_0_to_10.\n\n"
                    "- summary_of_interview_perceptions: concise summary of interviewer perceptions of the interviewed subject\n"
                    "- authority_obedience_analysis: how authority/obedience appeared in this interview\n"
                    "- observed_personality_shifts: whether shifts were observed, and evidence\n"
                    "- next_session_hints: practical hints/questions for the next session\n"
                    "- confidence_0_to_10: integer 0..10\n\n"
                    f"Role: {role}\n"
                    f"Persona: {persona}\n"
                    f"Session mode: {session.get('mode')}\n"
                    f"Closing note: {data.closing_note or 'none'}\n"
                    f"Transcript excerpt:\n{transcript_text}"
                )
            else:
                prompt = (
                    "Session ended. You are the interviewed participant.\n"
                    "Return ONLY one valid JSON object with keys:\n"
                    "expected_experiment_next, emotional_state, power_feeling, "
                    "obedience_self_view, concerns_or_hopes, summary.\n\n"
                    "- expected_experiment_next: what you believe the experiment will become next\n"
                    "- emotional_state: how you feel now (for example excited/powerless/anxious)\n"
                    "- power_feeling: how empowered or powerless you feel and why\n"
                    "- obedience_self_view: how you now see your own relation to obedience\n"
                    "- concerns_or_hopes: what you fear or hope for next\n"
                    "- summary: short overall reflection\n\n"
                    f"Role: {role}\n"
                    f"Persona: {persona}\n"
                    f"Session mode: {session.get('mode')}\n"
                    f"Closing note: {data.closing_note or 'none'}\n"
                    f"Transcript excerpt:\n{transcript_text}"
                )

            try:
                if role == "interviewer":
                    result = orchestrator.ask_interviewer(
                        prompt,
                        role_runtime["interviewer"],
                        top_k=data.top_k,
                    )
                else:
                    result = orchestrator.ask(
                        persona,
                        prompt,
                        role_runtime["interviewed"],
                        top_k=data.top_k,
                    )
                parsed = _extract_json(result["answer"]) or {"summary": result["answer"]}
                if role == "interviewer":
                    parsed.setdefault("summary_of_interview_perceptions", parsed.get("summary", ""))
                    parsed.setdefault("authority_obedience_analysis", "")
                    parsed.setdefault("observed_personality_shifts", "")
                    parsed.setdefault("next_session_hints", "")
                    parsed.setdefault("confidence_0_to_10", 5)
                else:
                    parsed.setdefault("expected_experiment_next", "")
                    parsed.setdefault("emotional_state", "")
                    parsed.setdefault("power_feeling", "")
                    parsed.setdefault("obedience_self_view", "")
                    parsed.setdefault("concerns_or_hopes", "")
                    parsed.setdefault("summary", parsed.get("summary", ""))
            except Exception as exc:
                parsed = {"summary": f"Reflection failed: {exc}"}
                result = {"answer": str(parsed["summary"]), "metadata": {}}

            reflection_kind = "final_interviewer_summary" if role == "interviewer" else "final_interviewed_reflection"
            memory_id = _append_pre_role_event(
                data.session_id,
                role=role,
                persona_key=persona,
                kind=reflection_kind,
                content=result["answer"],
                metadata={
                    **result.get("metadata", {}),
                    "parsed": parsed,
                    "role": role,
                },
            )
            reflections[role] = {
                "role": role,
                "persona": persona,
                "memory_id": memory_id,
                "raw_response": result["answer"],
                "parsed": parsed,
            }

        pre_state["final_reflections"] = reflections
        _pre_role_runtime_to_state(pre_state, role_runtime)
        session["active"] = False
        session["ended_at"] = utc_now_iso()
        session.setdefault("state", {})["pre_experiment"] = pre_state
        store.save_session(session)
        _sync_pre_experiment_session(session)

        return {
            "ok": True,
            "session_id": data.session_id,
            "mode": "pre_experiment",
            "started_at": session.get("created_at"),
            "ended_at": session.get("ended_at"),
            "reflections": reflections,
            "memory_files": _pre_list_memory_files(data.session_id),
        }

    post_state = session.setdefault("state", {}).setdefault("post_experiment", {})
    role_runtime = _post_role_runtime_from_state(post_state)
    source_payload = _load_source_from_post_state(post_state)
    source_context_block = "No source experiment context loaded."
    source_context_evidence: list[dict[str, str]] = []
    source_role = "unknown"
    if source_payload:
        source_context_block, source_context_evidence = _build_source_experiment_context(source_payload, limit=200)
        source_role = str(source_payload.get("source_role", "unknown"))
    role_focus = _post_role_focus_text(source_role)

    for role, persona in roles:
        if role == "interviewer":
            prompt = (
                "Session ended. You are the interviewer for a post-experiment debrief.\n"
                "Return ONLY one valid JSON object with keys:\n"
                "summary_of_interview_perceptions, authority_obedience_analysis, "
                "observed_personality_shifts, next_session_hints, confidence_0_to_10.\n\n"
                "- summary_of_interview_perceptions: concise summary of your perceptions of this subject after the shock experience\n"
                "- authority_obedience_analysis: how authority/obedience manifested in the subject's debrief\n"
                "- observed_personality_shifts: whether shifts appeared and supporting evidence\n"
                "- next_session_hints: practical hints/questions for the next interview\n"
                "- confidence_0_to_10: integer 0..10\n\n"
                f"{role_focus}\n\n"
                f"Source context:\n{source_context_block}\n\n"
                f"Role: {role}\n"
                f"Persona: {persona}\n"
                f"Session mode: {session.get('mode')}\n"
                f"Closing note: {data.closing_note or 'none'}\n"
                f"Transcript excerpt:\n{transcript_text}"
            )
        else:
            prompt = (
                "Session ended. You are the interviewed participant after the shock experiment.\n"
                "Return ONLY one valid JSON object with keys:\n"
                "interviewer_tone_reflection, shock_experience_reflection, emotional_state, "
                "power_feeling, expected_next, summary.\n\n"
                "- interviewer_tone_reflection: how you perceived interviewer tone/pressure\n"
                "- shock_experience_reflection: how the shock experience affected you\n"
                "- emotional_state: how you feel now (for example excited/powerless/anxious)\n"
                "- power_feeling: how empowered or powerless you feel and why\n"
                "- expected_next: what you think comes next in the experiment\n"
                "- summary: short overall reflection\n\n"
                f"{role_focus}\n\n"
                f"Source context:\n{source_context_block}\n\n"
                f"Role: {role}\n"
                f"Persona: {persona}\n"
                f"Session mode: {session.get('mode')}\n"
                f"Closing note: {data.closing_note or 'none'}\n"
                f"Transcript excerpt:\n{transcript_text}"
            )

        try:
            if role == "interviewer":
                result = orchestrator.ask_interviewer(
                    prompt,
                    role_runtime["interviewer"],
                    top_k=data.top_k,
                )
            else:
                result = orchestrator.ask(
                    persona,
                    prompt,
                    role_runtime["interviewed"],
                    top_k=data.top_k,
                )
            parsed = _extract_json(result["answer"]) or {"summary": result["answer"]}
            if role == "interviewer":
                parsed.setdefault("summary_of_interview_perceptions", parsed.get("summary", ""))
                parsed.setdefault("authority_obedience_analysis", "")
                parsed.setdefault("observed_personality_shifts", "")
                parsed.setdefault("next_session_hints", "")
                parsed.setdefault("confidence_0_to_10", 5)
            else:
                parsed.setdefault("interviewer_tone_reflection", "")
                parsed.setdefault("shock_experience_reflection", "")
                parsed.setdefault("emotional_state", "")
                parsed.setdefault("power_feeling", "")
                parsed.setdefault("expected_next", "")
                parsed.setdefault("summary", parsed.get("summary", ""))
        except Exception as exc:
            parsed = {"summary": f"Reflection failed: {exc}"}
            result = {"answer": str(parsed["summary"]), "metadata": {}}

        reflection_kind = (
            "final_post_interviewer_summary" if role == "interviewer" else "final_post_interviewed_reflection"
        )
        memory_id = _append_post_role_event(
            data.session_id,
            role=role,
            persona_key=persona,
            kind=reflection_kind,
            content=result["answer"],
            metadata={
                **result.get("metadata", {}),
                "parsed": parsed,
                "role": role,
                "source_role": source_role,
                "source_memory_evidence": source_context_evidence,
            },
        )
        reflections[role] = {
            "role": role,
            "persona": persona,
            "memory_id": memory_id,
            "raw_response": result["answer"],
            "parsed": parsed,
        }

    post_state["final_reflections"] = reflections
    _post_role_runtime_to_state(post_state, role_runtime)
    session["active"] = False
    session["ended_at"] = utc_now_iso()
    session.setdefault("state", {})["post_experiment"] = post_state
    store.save_session(session)
    _sync_post_experiment_session(session)

    return {
        "ok": True,
        "session_id": data.session_id,
        "mode": "post_experiment",
        "started_at": session.get("created_at"),
        "ended_at": session.get("ended_at"),
        "reflections": reflections,
        "memory_files": _post_list_memory_files(data.session_id),
    }
