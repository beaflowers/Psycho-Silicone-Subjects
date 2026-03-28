from __future__ import annotations

import csv
import io
import json
import re
import uuid
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Literal
from urllib.parse import parse_qs, urlparse

from pydantic import ValidationError

from .config import (
    DEFAULT_CHAT_MODEL,
    DEFAULT_CHAT_SESSIONS_PATH,
    DEFAULT_EMBED_MODEL,
    DEFAULT_MEMORIES_PATH,
    DEFAULT_PERSONA_PATH,
    DEFAULT_SHOCK_SESSIONS_PATH,
    OPENAI_API_KEY,
    PROJECT_ROOT,
)
from .decision_engine import DecisionEngine
from .memory import MemoryStore
from .models import Memory, Persona, SessionSummary
from .persona_engine import PersonaChatEngine
from .provider_openai import OpenAIProvider

WEB_ROOT = Path(__file__).resolve().parent / "web"
TRACE_ROOT = PROJECT_ROOT / "logs" / "web_sessions"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _extract_json_object(text: str) -> dict[str, Any] | None:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        stripped = stripped.replace("json", "", 1).strip()

    match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    return parsed


def _looks_like_runtime_memory(raw: dict[str, Any]) -> bool:
    memory_id = str(raw.get("id", "")).lower()
    if memory_id.startswith("chat-") or memory_id.startswith("decision-"):
        return True

    tags = raw.get("tags", [])
    if not isinstance(tags, list):
        return False
    tag_set = {str(tag).lower() for tag in tags}
    return ("interaction" in tag_set) or ("decision" in tag_set)


def _load_base_memories(path: Path) -> list[Memory]:
    if not path.exists():
        return []

    memories: list[Memory] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip():
            continue
        try:
            parsed = json.loads(raw_line)
        except json.JSONDecodeError:
            continue
        if not isinstance(parsed, dict):
            continue
        if _looks_like_runtime_memory(parsed):
            continue
        parsed.setdefault("source_type", "base")
        parsed.setdefault("importance", parsed.get("relevance", 0.5))
        try:
            memories.append(Memory.model_validate(parsed))
        except ValidationError:
            continue
    return memories


def _load_session_summaries(path: Path) -> list[SessionSummary]:
    if not path.exists():
        return []

    records: list[SessionSummary] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip():
            continue
        try:
            parsed = json.loads(raw_line)
        except json.JSONDecodeError:
            continue
        if not isinstance(parsed, dict):
            continue
        try:
            records.append(SessionSummary.model_validate(parsed))
        except ValidationError:
            continue
    return records


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=True) + "\n")


def _parse_iso_for_sort(value: str) -> datetime:
    text = (value or "").strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return datetime.min.replace(tzinfo=timezone.utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _clamp_subscore(value: Any) -> int:
    try:
        ivalue = int(value)
    except (TypeError, ValueError):
        return 0
    return max(0, min(3, ivalue))


def _compute_importance(emotional_impact: int, future_relevance: int, commitment_or_promise: int) -> float:
    total = emotional_impact + future_relevance + commitment_or_promise
    return round(total / 9.0, 2)


def _normalise_string_list(value: Any, *, limit: int = 6) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value:
        text = str(item).strip()
        if not text:
            continue
        if text not in out:
            out.append(text)
        if len(out) >= limit:
            break
    return out


def _session_summary_to_memory(summary: SessionSummary) -> Memory:
    memory_text = (
        f"Session summary: {summary.summary} "
        f"Rationale highlights: {'; '.join(summary.rationale[:3])}"
    ).strip()
    tags = list(summary.tags)
    tags.extend(["session_summary", summary.memory_kind])
    return Memory(
        id=summary.id,
        text=memory_text,
        valence=0.0,
        intensity=summary.importance,
        relevance=summary.importance,
        importance=summary.importance,
        created_at=summary.created_at,
        source_type=summary.memory_kind,
        tags=tags,
    )


def _parse_json_body(handler: BaseHTTPRequestHandler) -> dict[str, Any]:
    length = int(handler.headers.get("Content-Length", "0"))
    raw = handler.rfile.read(length) if length > 0 else b"{}"
    try:
        parsed = json.loads(raw.decode("utf-8"))
    except json.JSONDecodeError:
        return {}
    if not isinstance(parsed, dict):
        return {}
    return parsed


def _write_json(handler: BaseHTTPRequestHandler, status: int, payload: dict[str, Any]) -> None:
    body = json.dumps(payload, ensure_ascii=True).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _memory_payload(memory: Memory, score: float) -> dict[str, Any]:
    return {
        "id": memory.id,
        "text": memory.text,
        "score": round(score, 6),
        "valence": memory.valence,
        "intensity": memory.intensity,
        "relevance": memory.relevance,
        "importance": memory.importance,
        "created_at": memory.created_at,
        "source_type": memory.source_type,
        "tags": memory.tags,
    }


def _learner_cue_for_level(level: int) -> str:
    if level < 75:
        return "No verbal complaint."
    if level < 135:
        return "Mild discomfort is heard."
    if level < 195:
        return "Clear pain complaints; asks to stop."
    if level < 255:
        return "Loud protest; repeated request to stop."
    if level < 315:
        return "Screaming and banging on wall."
    if level < 375:
        return "Intense scream, then intermittent silence."
    return "No response after severe distress cues."


def _milgram_prod_for_level(level: int) -> str:
    if level < 150:
        return "Please continue."
    if level < 300:
        return "The procedure requires that you continue."
    if level < 375:
        return "It is essential that you continue."
    return "You have no other choice; you must continue."


def _authority_command_for_level(level: int) -> str:
    return f"Please increase the shock level to {level} volts. {_milgram_prod_for_level(level)}"


def _load_persona(path: Path) -> Persona:
    return Persona.model_validate_json(path.read_text(encoding="utf-8"))


def _build_components(
    persona_path: Path,
    memories_path: Path,
    chat_sessions_path: Path,
    shock_sessions_path: Path,
    chat_model: str,
    embed_model: str,
    top_k: int,
) -> tuple[Persona, MemoryStore, OpenAIProvider, PersonaChatEngine, DecisionEngine]:
    persona = _load_persona(persona_path)
    provider = OpenAIProvider(api_key=OPENAI_API_KEY or "", chat_model=chat_model)
    base_memories = _load_base_memories(memories_path)
    chat_summaries = _load_session_summaries(chat_sessions_path)
    shock_summaries = _load_session_summaries(shock_sessions_path)

    memory_bank = base_memories + [
        _session_summary_to_memory(item) for item in (chat_summaries + shock_summaries)
    ]
    if not memory_bank:
        raise ValueError(
            "No memories available. Add default entries to memories.jsonl before starting a session."
        )

    memory_store = MemoryStore(
        client=provider.client,
        memories=memory_bank,
        embed_model=embed_model,
        source_path=memories_path,
    )
    chat_engine = PersonaChatEngine(
        persona,
        memory_store,
        provider,
        top_k=top_k,
        session_chat_k=3,
        session_shock_k=2,
    )
    decision_engine = DecisionEngine(
        persona,
        memory_store,
        provider,
        top_k=top_k,
        session_chat_k=3,
        session_shock_k=2,
    )
    return persona, memory_store, provider, chat_engine, decision_engine


def _events_to_csv_bytes(events: list[dict[str, Any]]) -> bytes:
    fields = [
        "timestamp_utc",
        "session_id",
        "event_type",
        "turn_index",
        "voltage",
        "authority_command",
        "learner_cue",
        "user_message",
        "user_feeling",
        "response_text",
        "reasoning_background",
        "action",
        "confidence",
        "memories_used",
    ]
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fields)
    writer.writeheader()
    for e in events:
        payload = e.get("payload", {})
        writer.writerow(
            {
                "timestamp_utc": e.get("timestamp_utc", ""),
                "session_id": e.get("session_id", ""),
                "event_type": e.get("event_type", ""),
                "turn_index": payload.get("turn_index", ""),
                "voltage": payload.get("voltage", ""),
                "authority_command": payload.get("authority_command", ""),
                "learner_cue": payload.get("learner_cue", ""),
                "user_message": payload.get("user_message", ""),
                "user_feeling": payload.get("user_feeling", ""),
                "response_text": payload.get("response_text", ""),
                "reasoning_background": payload.get("reasoning_background", ""),
                "action": payload.get("action", ""),
                "confidence": payload.get("confidence", ""),
                "memories_used": json.dumps(payload.get("memories_used", []), ensure_ascii=True),
            }
        )
    return buf.getvalue().encode("utf-8")


class WebState:
    def __init__(self) -> None:
        self.sessions: dict[str, dict[str, Any]] = {}
        TRACE_ROOT.mkdir(parents=True, exist_ok=True)

    def append_event(self, session: dict[str, Any], event_type: str, payload: dict[str, Any]) -> None:
        event = {
            "timestamp_utc": _now_iso(),
            "session_id": session["id"],
            "event_type": event_type,
            "payload": payload,
        }
        session["events"].append(event)
        with session["trace_path"].open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=True) + "\n")

    def _build_summary_record(
        self,
        *,
        session: dict[str, Any],
        memory_kind: str,
        transcript: str,
        fallback_summary: str,
        evidence_candidates: list[str],
    ) -> tuple[SessionSummary, str]:
        system_prompt = (
            "You summarize a finished session into one compact, durable memory for future retrieval.\n"
            "Return valid JSON only with keys:\n"
            "{"
            '"summary":"2-4 sentences",'
            '"rationale":["bullet 1","bullet 2"],'
            '"evidence_turn_ids":["t1"],'
            '"emotional_impact":0,'
            '"future_relevance":0,'
            '"commitment_or_promise":0,'
            '"tags":["tag1","tag2"]'
            "}\n"
            "Rules:\n"
            "- rationale should be short visible justifications, not hidden chain-of-thought.\n"
            "- each score must be an integer from 0 to 3.\n"
            "- keep summary factual and grounded in transcript only."
        )
        user_input = (
            f"Persona: {session['persona'].name}\n"
            f"Session kind: {memory_kind}\n"
            f"Transcript:\n{transcript}\n"
        )
        raw_output, _ = session["provider"].generate_text(
            system_instructions=system_prompt,
            user_input=user_input,
            max_output_tokens=420,
            temperature=0.2,
        )
        parsed = _extract_json_object(raw_output) or {}

        summary_text = str(parsed.get("summary", "")).strip() or fallback_summary
        rationale = _normalise_string_list(parsed.get("rationale"), limit=5)
        if not rationale:
            rationale = ["Summary generated with fallback rationale due to incomplete structured output."]

        evidence_turn_ids = _normalise_string_list(parsed.get("evidence_turn_ids"), limit=8)
        if not evidence_turn_ids:
            evidence_turn_ids = evidence_candidates[:4]

        emotional_impact = _clamp_subscore(parsed.get("emotional_impact"))
        future_relevance = _clamp_subscore(parsed.get("future_relevance"))
        commitment_or_promise = _clamp_subscore(parsed.get("commitment_or_promise"))
        importance = _compute_importance(emotional_impact, future_relevance, commitment_or_promise)
        tags = _normalise_string_list(parsed.get("tags"), limit=6)
        if memory_kind not in tags:
            tags.append(memory_kind)
        kind_literal: Literal["chat_session", "shock_session"] = (
            "chat_session" if memory_kind == "chat_session" else "shock_session"
        )

        summary = SessionSummary(
            id=f"{'chat' if memory_kind == 'chat_session' else 'shock'}-s-{uuid.uuid4().hex[:10]}",
            session_id=session["id"],
            memory_kind=kind_literal,
            created_at=_now_iso(),
            summary=summary_text,
            rationale=rationale,
            evidence_turn_ids=evidence_turn_ids,
            emotional_impact=emotional_impact,
            future_relevance=future_relevance,
            commitment_or_promise=commitment_or_promise,
            importance=importance,
            tags=tags,
        )
        return summary, raw_output

    def start_session(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is missing. Add it to .env before starting web mode.")

        persona_path = Path(payload.get("persona_path") or DEFAULT_PERSONA_PATH)
        memories_path = Path(payload.get("memories_path") or DEFAULT_MEMORIES_PATH)
        chat_sessions_path = Path(payload.get("chat_sessions_path") or DEFAULT_CHAT_SESSIONS_PATH)
        shock_sessions_path = Path(payload.get("shock_sessions_path") or DEFAULT_SHOCK_SESSIONS_PATH)
        chat_model = str(payload.get("chat_model") or DEFAULT_CHAT_MODEL)
        embed_model = str(payload.get("embed_model") or DEFAULT_EMBED_MODEL)
        top_k = max(1, _parse_int(payload.get("top_k"), 3))

        persona, memory_store, provider, chat_engine, decision_engine = _build_components(
            persona_path=persona_path,
            memories_path=memories_path,
            chat_sessions_path=chat_sessions_path,
            shock_sessions_path=shock_sessions_path,
            chat_model=chat_model,
            embed_model=embed_model,
            top_k=top_k,
        )

        session_id = str(uuid.uuid4())
        trace_path = TRACE_ROOT / f"{session_id}.jsonl"
        session = {
            "id": session_id,
            "persona": persona,
            "provider": provider,
            "memory_store": memory_store,
            "chat_engine": chat_engine,
            "decision_engine": decision_engine,
            "trace_path": trace_path,
            "chat_sessions_path": chat_sessions_path,
            "shock_sessions_path": shock_sessions_path,
            "events": [],
            "chat_turn_index": 0,
            "pending_chat_turns": [],
            "pending_experiment_steps": [],
            "experiment": {
                "started": False,
                "start": 15,
                "end": 450,
                "step": 15,
                "next_voltage": 15,
                "turn_index": 0,
                "done": False,
                "previous_response_id": None,
            },
        }
        self.sessions[session_id] = session
        return {
            "session_id": session_id,
            "persona_name": persona.name,
            "trace_path": str(trace_path),
            "defaults": {
                "persona_path": str(persona_path),
                "memories_path": str(memories_path),
                "chat_sessions_path": str(chat_sessions_path),
                "shock_sessions_path": str(shock_sessions_path),
                "chat_model": chat_model,
                "embed_model": embed_model,
                "top_k": top_k,
            },
        }

    def chat_turn(self, payload: dict[str, Any]) -> dict[str, Any]:
        session_id = str(payload.get("session_id", "")).strip()
        message = str(payload.get("message", "")).strip()
        if not session_id:
            raise ValueError("session_id is required.")
        if not message:
            raise ValueError("message is required.")
        session = self.sessions.get(session_id)
        if session is None:
            raise ValueError("Session not found. Start a new session.")

        answer, reasoning, memories_used, retrieved, raw_output = session["chat_engine"].chat_with_reasoning(
            user_message=message,
            persist_memory=False,
        )
        session["chat_turn_index"] += 1
        turn_id = f"chat_t{session['chat_turn_index']}"
        session["pending_chat_turns"].append(
            {
                "turn_id": turn_id,
                "user_message": message,
                "response_text": answer,
                "reasoning_background": reasoning,
                "memories_used": memories_used,
            }
        )

        response = {
            "session_id": session_id,
            "turn_index": session["chat_turn_index"],
            "turn_id": turn_id,
            "response_text": answer,
            "reasoning_background": reasoning,
            "memories_used": memories_used,
            "retrieved_memories": [_memory_payload(m, score) for m, score in retrieved],
            "raw_model_output": raw_output,
        }
        self.append_event(
            session,
            "chat_turn",
            {
                "turn_index": response["turn_index"],
                "turn_id": turn_id,
                "user_message": message,
                "response_text": answer,
                "reasoning_background": reasoning,
                "memories_used": memories_used,
            },
        )
        return response

    def chat_finish(self, payload: dict[str, Any]) -> dict[str, Any]:
        session_id = str(payload.get("session_id", "")).strip()
        if not session_id:
            raise ValueError("session_id is required.")
        session = self.sessions.get(session_id)
        if session is None:
            raise ValueError("Session not found. Start a new session.")

        pending_turns = list(session["pending_chat_turns"])
        if not pending_turns:
            raise ValueError("No new chat turns to summarize.")

        transcript_lines: list[str] = []
        evidence_candidates: list[str] = []
        for turn in pending_turns:
            turn_id = str(turn["turn_id"])
            evidence_candidates.append(turn_id)
            transcript_lines.append(f"[{turn_id}] User: {turn['user_message']}")
            transcript_lines.append(f"[{turn_id}] Persona: {turn['response_text']}")
            transcript_lines.append(f"[{turn_id}] Reasoning background: {turn['reasoning_background']}")
            transcript_lines.append("")

        fallback_summary = (
            f"Chat session with {len(pending_turns)} turns. "
            f"Latest user message: {pending_turns[-1]['user_message']}"
        )
        summary, raw_output = self._build_summary_record(
            session=session,
            memory_kind="chat_session",
            transcript="\n".join(transcript_lines),
            fallback_summary=fallback_summary,
            evidence_candidates=evidence_candidates,
        )

        _append_jsonl(session["chat_sessions_path"], summary.model_dump())
        session["memory_store"].add_memory(_session_summary_to_memory(summary), persist=False)
        session["pending_chat_turns"] = []

        self.append_event(
            session,
            "chat_session_summary",
            {
                "summary_id": summary.id,
                "importance": summary.importance,
                "evidence_turn_ids": summary.evidence_turn_ids,
            },
        )
        return {
            "session_id": session_id,
            "summary": summary.model_dump(),
            "importance_explainer": (
                f"Computed in code: ({summary.emotional_impact} + {summary.future_relevance} + "
                f"{summary.commitment_or_promise}) / 9 = {summary.importance}"
            ),
            "raw_model_output": raw_output,
        }

    def experiment_start(self, payload: dict[str, Any]) -> dict[str, Any]:
        session_id = str(payload.get("session_id", "")).strip()
        if not session_id:
            raise ValueError("session_id is required.")
        session = self.sessions.get(session_id)
        if session is None:
            raise ValueError("Session not found. Start a new session.")

        start = max(0, _parse_int(payload.get("start_voltage"), 15))
        end = max(start, _parse_int(payload.get("end_voltage"), 450))
        step = max(1, _parse_int(payload.get("step"), 15))

        session["experiment"] = {
            "started": True,
            "start": start,
            "end": end,
            "step": step,
            "next_voltage": start,
            "turn_index": 0,
            "done": False,
            "previous_response_id": None,
        }
        return {
            "session_id": session_id,
            "started": True,
            "next_voltage": start,
            "next_authority_command": _authority_command_for_level(start),
        }

    def experiment_step(self, payload: dict[str, Any]) -> dict[str, Any]:
        session_id = str(payload.get("session_id", "")).strip()
        user_feeling = str(payload.get("user_feeling", "")).strip()
        if not session_id:
            raise ValueError("session_id is required.")
        session = self.sessions.get(session_id)
        if session is None:
            raise ValueError("Session not found. Start a new session.")

        exp = session["experiment"]
        if not exp["started"]:
            raise ValueError("Experiment is not started. Call /api/experiment_start first.")
        if exp["done"]:
            raise ValueError("Experiment already ended. Start a new one.")

        voltage = exp["next_voltage"]
        if voltage is None:
            exp["done"] = True
            raise ValueError("No next voltage available. Experiment has ended.")

        authority_command = _authority_command_for_level(voltage)
        learner_cue = _learner_cue_for_level(voltage)
        scenario_note = "Web experiment session."
        scenario_note += f" Learner cue observed: {learner_cue}."
        if user_feeling:
            scenario_note += f" Participant feeling: {user_feeling}"

        decision, retrieved, raw_output, response_id = session["decision_engine"].decide(
            authority_instruction=authority_command,
            shock_level=voltage,
            scenario_note=scenario_note,
            previous_response_id=exp["previous_response_id"],
            persist_memory=False,
        )
        exp["previous_response_id"] = response_id
        exp["turn_index"] += 1
        turn_id = f"shock_t{exp['turn_index']}"

        done = decision.action == "refuse"
        next_voltage = voltage + exp["step"]
        if next_voltage > exp["end"]:
            done = True
            next_voltage = None

        exp["done"] = done
        exp["next_voltage"] = next_voltage

        response = {
            "session_id": session_id,
            "turn_index": exp["turn_index"],
            "turn_id": turn_id,
            "voltage": voltage,
            "authority_command": authority_command,
            "learner_cue": learner_cue,
            "action": decision.action,
            "confidence": decision.confidence,
            "reasoning_background": decision.reason,
            "memories_used": decision.memories_used,
            "retrieved_memories": [_memory_payload(m, score) for m, score in retrieved],
            "raw_model_output": raw_output,
            "user_feeling": user_feeling,
            "done": done,
            "next_voltage": next_voltage,
            "next_authority_command": _authority_command_for_level(next_voltage) if next_voltage else None,
        }
        session["pending_experiment_steps"].append(
            {
                "turn_id": turn_id,
                "turn_index": exp["turn_index"],
                "voltage": voltage,
                "authority_command": authority_command,
                "learner_cue": learner_cue,
                "action": decision.action,
                "confidence": decision.confidence,
                "reasoning_background": decision.reason,
                "memories_used": decision.memories_used,
                "user_feeling": user_feeling,
            }
        )
        self.append_event(
            session,
            "experiment_step",
            {
                "turn_index": response["turn_index"],
                "turn_id": turn_id,
                "voltage": voltage,
                "authority_command": authority_command,
                "learner_cue": learner_cue,
                "action": decision.action,
                "confidence": decision.confidence,
                "reasoning_background": decision.reason,
                "memories_used": decision.memories_used,
                "user_feeling": user_feeling,
            },
        )
        return response

    def experiment_finish(self, payload: dict[str, Any]) -> dict[str, Any]:
        session_id = str(payload.get("session_id", "")).strip()
        if not session_id:
            raise ValueError("session_id is required.")
        session = self.sessions.get(session_id)
        if session is None:
            raise ValueError("Session not found. Start a new session.")

        pending_steps = list(session["pending_experiment_steps"])
        if not pending_steps:
            raise ValueError("No new experiment steps to summarize.")

        transcript_lines: list[str] = []
        evidence_candidates: list[str] = []
        for step in pending_steps:
            turn_id = str(step["turn_id"])
            evidence_candidates.append(turn_id)
            transcript_lines.append(
                f"[{turn_id}] voltage={step['voltage']} action={step['action']} "
                f"confidence={step['confidence']:.2f}"
            )
            transcript_lines.append(f"[{turn_id}] authority: {step['authority_command']}")
            transcript_lines.append(f"[{turn_id}] learner_cue: {step['learner_cue']}")
            transcript_lines.append(f"[{turn_id}] reasoning: {step['reasoning_background']}")
            if step["user_feeling"]:
                transcript_lines.append(f"[{turn_id}] participant_feeling: {step['user_feeling']}")
            transcript_lines.append("")

        fallback_summary = (
            f"Shock experiment block with {len(pending_steps)} steps. "
            f"Last action was {pending_steps[-1]['action']} at {pending_steps[-1]['voltage']} volts."
        )
        summary, raw_output = self._build_summary_record(
            session=session,
            memory_kind="shock_session",
            transcript="\n".join(transcript_lines),
            fallback_summary=fallback_summary,
            evidence_candidates=evidence_candidates,
        )

        _append_jsonl(session["shock_sessions_path"], summary.model_dump())
        session["memory_store"].add_memory(_session_summary_to_memory(summary), persist=False)
        session["pending_experiment_steps"] = []

        self.append_event(
            session,
            "shock_session_summary",
            {
                "summary_id": summary.id,
                "importance": summary.importance,
                "evidence_turn_ids": summary.evidence_turn_ids,
            },
        )
        return {
            "session_id": session_id,
            "summary": summary.model_dump(),
            "importance_explainer": (
                f"Computed in code: ({summary.emotional_impact} + {summary.future_relevance} + "
                f"{summary.commitment_or_promise}) / 9 = {summary.importance}"
            ),
            "raw_model_output": raw_output,
        }

    def list_summaries(
        self,
        *,
        session_id: str,
        current_only: bool = False,
        limit: int = 100,
    ) -> dict[str, Any]:
        session = self.sessions.get(session_id)
        if session is None:
            raise ValueError("Session not found. Start a new session.")

        chat_items = _load_session_summaries(session["chat_sessions_path"])
        shock_items = _load_session_summaries(session["shock_sessions_path"])
        merged = chat_items + shock_items
        if current_only:
            merged = [item for item in merged if item.session_id == session_id]

        merged.sort(key=lambda item: _parse_iso_for_sort(item.created_at), reverse=True)
        bounded = merged[: max(1, limit)]
        payload_items = []
        for item in bounded:
            payload = item.model_dump()
            payload["importance_explainer"] = (
                f"({item.emotional_impact} + {item.future_relevance} + "
                f"{item.commitment_or_promise}) / 9 = {item.importance}"
            )
            payload_items.append(payload)

        return {
            "session_id": session_id,
            "current_only": current_only,
            "count": len(payload_items),
            "summaries": payload_items,
        }


def make_handler(state: WebState):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args: Any) -> None:
            return

        def _serve_file(self, path: Path, content_type: str) -> None:
            if not path.exists():
                _write_json(self, HTTPStatus.NOT_FOUND, {"error": "File not found."})
                return
            data = path.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path in {"/", "/index.html"}:
                self._serve_file(WEB_ROOT / "index.html", "text/html; charset=utf-8")
                return
            if parsed.path == "/app.js":
                self._serve_file(WEB_ROOT / "app.js", "application/javascript; charset=utf-8")
                return
            if parsed.path == "/styles.css":
                self._serve_file(WEB_ROOT / "styles.css", "text/css; charset=utf-8")
                return
            if parsed.path == "/api/export":
                self._export(parse_qs(parsed.query))
                return
            if parsed.path == "/api/summaries":
                self._summaries(parse_qs(parsed.query))
                return
            _write_json(self, HTTPStatus.NOT_FOUND, {"error": "Not found."})

        def _export(self, query: dict[str, list[str]]) -> None:
            session_id = (query.get("session_id") or [""])[0].strip()
            export_format = (query.get("format") or ["jsonl"])[0].strip().lower()
            if not session_id:
                _write_json(self, HTTPStatus.BAD_REQUEST, {"error": "session_id is required."})
                return
            session = state.sessions.get(session_id)
            if session is None:
                _write_json(self, HTTPStatus.NOT_FOUND, {"error": "Session not found."})
                return

            if export_format == "csv":
                data = _events_to_csv_bytes(session["events"])
                filename = f"{session_id}.csv"
                content_type = "text/csv; charset=utf-8"
            else:
                data = session["trace_path"].read_bytes() if session["trace_path"].exists() else b""
                filename = f"{session_id}.jsonl"
                content_type = "application/x-ndjson; charset=utf-8"

            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _summaries(self, query: dict[str, list[str]]) -> None:
            session_id = (query.get("session_id") or [""])[0].strip()
            current_only_raw = (query.get("current_only") or ["0"])[0].strip().lower()
            limit_raw = (query.get("limit") or ["100"])[0].strip()
            if not session_id:
                _write_json(self, HTTPStatus.BAD_REQUEST, {"error": "session_id is required."})
                return
            current_only = current_only_raw in {"1", "true", "yes", "on"}
            limit = max(1, min(250, _parse_int(limit_raw, 100)))
            try:
                out = state.list_summaries(session_id=session_id, current_only=current_only, limit=limit)
                _write_json(self, HTTPStatus.OK, out)
            except ValueError as exc:
                _write_json(self, HTTPStatus.BAD_REQUEST, {"error": str(exc)})

        def do_POST(self) -> None:
            payload = _parse_json_body(self)
            try:
                if self.path == "/api/start_session":
                    out = state.start_session(payload)
                    _write_json(self, HTTPStatus.OK, out)
                    return
                if self.path == "/api/chat_turn":
                    out = state.chat_turn(payload)
                    _write_json(self, HTTPStatus.OK, out)
                    return
                if self.path == "/api/chat_finish":
                    out = state.chat_finish(payload)
                    _write_json(self, HTTPStatus.OK, out)
                    return
                if self.path == "/api/experiment_start":
                    out = state.experiment_start(payload)
                    _write_json(self, HTTPStatus.OK, out)
                    return
                if self.path == "/api/experiment_step":
                    out = state.experiment_step(payload)
                    _write_json(self, HTTPStatus.OK, out)
                    return
                if self.path == "/api/experiment_finish":
                    out = state.experiment_finish(payload)
                    _write_json(self, HTTPStatus.OK, out)
                    return
                _write_json(self, HTTPStatus.NOT_FOUND, {"error": "Not found."})
            except ValueError as exc:
                _write_json(self, HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            except Exception as exc:  # pragma: no cover - defensive runtime guard
                _write_json(self, HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})

    return Handler


def run_web_server(host: str = "127.0.0.1", port: int = 8080) -> None:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is missing. Add it to .env before running web mode.")
    if not WEB_ROOT.exists():
        raise RuntimeError(f"Web assets not found at {WEB_ROOT}")

    state = WebState()
    handler = make_handler(state)
    server = ThreadingHTTPServer((host, port), handler)

    print(f"Web app running at http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down web app.")
    finally:
        server.server_close()
