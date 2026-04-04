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

from milgram_experiment.backend.persona_adapters import (
    PERSONA_FEMWIFE,
    PERSONA_JEKYLL,
    PersonaOrchestrator,
    PersonaRuntimeState,
)
from milgram_experiment.backend.session_store import JsonSessionStore, utc_now_iso


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
APP_ROOT = Path(__file__).resolve().parents[1]
WEB_DIR = APP_ROOT / "web"
DATA_DIR = APP_ROOT / "data"

PERSONA_LABELS = {
    PERSONA_JEKYLL: "Jekyll/Hyde",
    PERSONA_FEMWIFE: "Femwife (Pico)",
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

store = JsonSessionStore(DATA_DIR)
_orchestrator: PersonaOrchestrator | None = None


class StartSessionRequest(BaseModel):
    chat_persona: str = Field(default=PERSONA_JEKYLL, min_length=1, max_length=80)
    context: str = Field(default="", max_length=6000)
    top_k: int = Field(default=5, ge=1, le=20)


class ChatTurnRequest(BaseModel):
    session_id: str = Field(min_length=3, max_length=120)
    message: str = Field(min_length=1, max_length=6000)
    top_k: int = Field(default=5, ge=1, le=20)


class FinishSessionRequest(BaseModel):
    session_id: str = Field(min_length=3, max_length=120)
    closing_note: str = Field(default="", max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)


app = FastAPI(title="Subject Chat")
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


def _new_session_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def _runtime_from_session(session: dict[str, Any]) -> dict[str, PersonaRuntimeState]:
    runtime_payload = session.setdefault("state", {}).setdefault("persona_runtime", {})
    runtime: dict[str, PersonaRuntimeState] = {}

    for persona_key in session.get("participants", []):
        payload = runtime_payload.get(persona_key, {})
        runtime[persona_key] = PersonaRuntimeState(
            previous_response_id=payload.get("previous_response_id"),
            shift=float(payload.get("shift", 0.5)),
        )
    return runtime


def _runtime_to_session(session: dict[str, Any], runtime: dict[str, PersonaRuntimeState]) -> None:
    session.setdefault("state", {})["persona_runtime"] = {
        key: {
            "previous_response_id": value.previous_response_id,
            "shift": value.shift,
        }
        for key, value in runtime.items()
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


def _append_persona_event(
    session_id: str,
    persona_key: str,
    kind: str,
    content: str,
    *,
    metadata: dict[str, Any] | None = None,
) -> str:
    memory_id = store.append_persona_memory(
        session_id,
        persona_key,
        kind,
        content,
        metadata=metadata,
    )
    store.append_transcript(
        session_id,
        {
            "type": "persona_output",
            "persona": persona_key,
            "kind": kind,
            "memory_id": memory_id,
            "content": content,
            "metadata": metadata or {},
        },
    )
    return memory_id


def _extract_json(text: str) -> dict[str, Any] | None:
    raw = text.strip()
    if not raw:
        return None

    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        return None

    try:
        parsed = json.loads(match.group(0))
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        return None
    return None


def _build_recent_memory_block(session_id: str, persona_key: str, limit: int = 3) -> str:
    entries = store.get_persona_memories(session_id, persona_key)
    if not entries:
        return ""
    recent = entries[-limit:]
    lines = [f"- ({item['memory_id']}) {item['content']}" for item in recent]
    return "\n".join(lines)


def _render_transcript(session: dict[str, Any], limit: int = 40) -> str:
    rows = session.get("transcript", [])[-limit:]
    rendered: list[str] = []
    for row in rows:
        row_type = row.get("type")
        if row_type == "user_input":
            rendered.append(f"USER: {row.get('message', '')}")
            continue
        if row_type == "persona_output":
            persona = row.get("persona", "unknown")
            rendered.append(f"{persona.upper()}: {row.get('content', '')}")
    return "\n".join(rendered)


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
        sources = orchestrator.describe_sources()
        return {
            "ok": True,
            "sources": sources,
            "personas": PERSONA_LABELS,
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


@app.post("/api/session/start")
def start_session(data: StartSessionRequest) -> dict[str, Any]:
    chat_persona = _normalize_persona(data.chat_persona, fallback=PERSONA_JEKYLL)
    participants = [chat_persona]
    session_id = _new_session_id("chat")
    now = utc_now_iso()

    session = {
        "session_id": session_id,
        "mode": "chat",
        "active": True,
        "participants": participants,
        "config": {
            "chat_persona": chat_persona,
            "top_k": data.top_k,
            "context": data.context,
        },
        "created_at": now,
        "updated_at": now,
        "ended_at": None,
        "state": {
            "persona_runtime": {
                chat_persona: {
                    "previous_response_id": None,
                    "shift": 0.5,
                }
            },
        },
        "transcript": [],
    }

    store.create_session(session)
    store.ensure_persona_memory_file(session_id, chat_persona, "chat")

    return {
        "ok": True,
        "session_id": session_id,
        "mode": "chat",
        "participants": participants,
        "memory_files": store.list_session_memory_files(session_id),
    }


@app.get("/api/session/{session_id}")
def get_session(session_id: str) -> dict[str, Any]:
    try:
        session = store.load_session(session_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return {
        "ok": True,
        "session": session,
        "memory_files": store.list_session_memory_files(session_id),
    }


@app.post("/api/chat/turn")
def chat_turn(data: ChatTurnRequest) -> dict[str, Any]:
    try:
        session = store.load_session(data.session_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    if not session.get("active"):
        raise HTTPException(status_code=400, detail="Session is already finished.")
    if session.get("mode") != "chat":
        raise HTTPException(status_code=400, detail="Session is not in chat mode.")

    persona_key = session["config"]["chat_persona"]
    runtime = _runtime_from_session(session)
    orchestrator = _get_orchestrator()

    memory_block = _build_recent_memory_block(data.session_id, persona_key)
    prompt_parts = [
        "Chat session context for your continuity:",
        session["config"].get("context", "") or "No custom context.",
        f"User message: {data.message}",
    ]
    if memory_block:
        prompt_parts.append("Recent memory fragments:\n" + memory_block)
    prompt = "\n\n".join(prompt_parts)

    _append_user_event(data.session_id, data.message, "chat")

    try:
        result = orchestrator.ask(
            persona_key,
            prompt,
            runtime[persona_key],
            top_k=data.top_k,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    _runtime_to_session(session, runtime)
    store.save_session(session)

    memory_id = _append_persona_event(
        data.session_id,
        persona_key,
        "chat_reply",
        result["answer"],
        metadata=result.get("metadata"),
    )

    return {
        "ok": True,
        "session_id": data.session_id,
        "persona": persona_key,
        "reply": result["answer"],
        "memory_id": memory_id,
        "metadata": result.get("metadata", {}),
    }


@app.post("/api/session/finish")
def finish_session(data: FinishSessionRequest) -> dict[str, Any]:
    try:
        session = store.load_session(data.session_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    participants = list(session.get("participants", []))
    if not participants:
        raise HTTPException(status_code=400, detail="Session has no participants.")

    runtime = _runtime_from_session(session)
    orchestrator = _get_orchestrator()

    transcript_text = _render_transcript(session, limit=60)
    if data.closing_note.strip():
        _append_user_event(
            data.session_id,
            f"Session finish note: {data.closing_note}",
            session.get("mode", ""),
        )

    reflections: dict[str, Any] = {}
    for persona in participants:
        prompt = (
            "Session ended. Reflect over the whole interaction and return ONLY JSON with keys: "
            "summary, feelings, analysis_of_user, trauma_level_0_to_10, power_feeling_0_to_10.\n\n"
            f"Session mode: {session.get('mode')}\n"
            f"Closing note: {data.closing_note or 'none'}\n"
            f"Transcript excerpt:\n{transcript_text}"
        )

        try:
            result = orchestrator.ask(persona, prompt, runtime[persona], top_k=data.top_k)
            parsed = _extract_json(result["answer"]) or {"summary": result["answer"]}
        except Exception as exc:
            parsed = {"summary": f"Reflection failed: {exc}"}
            result = {"answer": str(parsed["summary"]), "metadata": {}}

        memory_id = _append_persona_event(
            data.session_id,
            persona,
            "final_reflection",
            result["answer"],
            metadata={
                **result.get("metadata", {}),
                "parsed": parsed,
            },
        )
        reflections[persona] = {
            "memory_id": memory_id,
            "raw_response": result["answer"],
            "parsed": parsed,
        }

    session["active"] = False
    session["ended_at"] = utc_now_iso()
    session.setdefault("state", {})["final_reflections"] = reflections

    _runtime_to_session(session, runtime)
    store.save_session(session)

    return {
        "ok": True,
        "session_id": data.session_id,
        "reflections": reflections,
        "memory_files": store.list_session_memory_files(data.session_id),
    }
