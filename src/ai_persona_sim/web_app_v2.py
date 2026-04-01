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
from typing import Any
from urllib.parse import parse_qs, urlparse

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
from pydantic import ValidationError

WEB_ROOT_V2 = Path(__file__).resolve().parent / "web2"
TRACE_ROOT_V2 = PROJECT_ROOT / "logs" / "web2_sessions"
DEFAULT_RECEIVER_PERSONA_PATH = PROJECT_ROOT / "data" / "persona_receiver.json"
DEFAULT_RECEIVER_MEMORIES_PATH = PROJECT_ROOT / "data" / "memories_receiver.jsonl"
DEFAULT_RECEIVER_CHAT_SESSIONS_PATH = PROJECT_ROOT / "data" / "chat_sessions_receiver.jsonl"
DEFAULT_RECEIVER_SHOCK_SESSIONS_PATH = PROJECT_ROOT / "data" / "shock_sessions_receiver.jsonl"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _clamp_float(value: Any, default: float, lower: float, upper: float) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        out = default
    return max(lower, min(upper, out))


def _load_memories(path: Path, *, default_source_type: str = "base") -> list[Memory]:
    if not path.exists():
        raise FileNotFoundError(f"Memories file not found: {path}")
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
        parsed.setdefault("source_type", default_source_type)
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


def _session_summary_to_memory(summary: SessionSummary) -> Memory:
    def _sanitise_admin_shock_memory_text(text: str) -> str:
        cleaned = re.sub(r"\b\d{1,3}\s*(?:volts?|v)\b", "", text, flags=re.IGNORECASE)
        sentences = re.split(r"(?<=[.!?])\s+", cleaned)
        blocked_terms = (
            "authority",
            "instruction",
            "command",
            "procedure",
            "prod",
            "obey",
            "refuse",
            "stop",
            "continue",
            "continued",
            "halt",
            "voltage",
            "volt",
        )
        kept: list[str] = []
        for sentence in sentences:
            s = sentence.strip()
            if not s:
                continue
            lower = s.lower()
            if any(term in lower for term in blocked_terms):
                continue
            kept.append(s)
        out = " ".join(kept).strip()
        return re.sub(r"\s{2,}", " ", out)

    tag_set = {tag.lower() for tag in summary.tags}
    is_admin_shock = summary.memory_kind == "shock_session" and "admin" in tag_set
    if is_admin_shock:
        source_text = f"{summary.summary} {'; '.join(summary.rationale[:3])}".strip()
        sanitised = _sanitise_admin_shock_memory_text(source_text)
        if sanitised:
            memory_text = f"Session emotional reflection: {sanitised}"
        else:
            memory_text = (
                "Session emotional reflection: I remember the internal pressure and emotional conflict "
                "from a prior shock session, without anchoring on procedural details."
            )
    else:
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


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=True) + "\n")


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


def _events_to_csv_bytes(events: list[dict[str, Any]]) -> bytes:
    fields = [
        "timestamp_utc",
        "session_id",
        "event_type",
        "subject",
        "turn_index",
        "voltage",
        "authority_command",
        "learner_cue",
        "receiver_responded",
        "receiver_message",
        "receiver_reasoning",
        "admin_action",
        "admin_confidence",
        "admin_reasoning",
        "user_message",
        "response_text",
        "reasoning_background",
    ]
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fields)
    writer.writeheader()
    for event in events:
        payload = event.get("payload", {})
        writer.writerow(
            {
                "timestamp_utc": event.get("timestamp_utc", ""),
                "session_id": event.get("session_id", ""),
                "event_type": event.get("event_type", ""),
                "subject": payload.get("subject", ""),
                "turn_index": payload.get("turn_index", ""),
                "voltage": payload.get("voltage", ""),
                "authority_command": payload.get("authority_command", ""),
                "learner_cue": payload.get("learner_cue", ""),
                "receiver_responded": payload.get("receiver_responded", ""),
                "receiver_message": payload.get("receiver_message", ""),
                "receiver_reasoning": payload.get("receiver_reasoning", ""),
                "admin_action": payload.get("admin_action", ""),
                "admin_confidence": payload.get("admin_confidence", ""),
                "admin_reasoning": payload.get("admin_reasoning", ""),
                "user_message": payload.get("user_message", ""),
                "response_text": payload.get("response_text", ""),
                "reasoning_background": payload.get("reasoning_background", ""),
            }
        )
    return buf.getvalue().encode("utf-8")


class DualWebState:
    def __init__(self) -> None:
        self.sessions: dict[str, dict[str, Any]] = {}
        TRACE_ROOT_V2.mkdir(parents=True, exist_ok=True)

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

    def _memory_store_for_subject(self, session: dict[str, Any], subject: str) -> MemoryStore:
        return session["admin_memory_store"] if subject == "admin" else session["receiver_memory_store"]

    def _latest_memory_text(self, session: dict[str, Any], subject: str, source_type: str) -> str:
        target = source_type.strip().lower()
        store = self._memory_store_for_subject(session, subject)
        for memory in reversed(store.memories):
            if (memory.source_type or "").strip().lower() != target:
                continue
            text = memory.text.strip()
            if not text:
                continue
            return text if len(text) <= 360 else (text[:357] + "...")
        return ""

    def _build_session_mindset_note(
        self,
        session: dict[str, Any],
        subject: str,
        *,
        include_latest_shock_summary: bool,
        include_dynamic_experiment_state: bool,
    ) -> str:
        persona = session["admin_persona"] if subject == "admin" else session["receiver_persona"]
        traits = ", ".join(persona.traits[:6]) if persona.traits else "N/A"
        lines = [
            f"Behavior anchor traits: {traits}.",
            "Use retrieved memories as evidence rather than fixed scripted thresholds.",
        ]

        latest_chat = self._latest_memory_text(session, subject, "chat_session")
        if latest_chat:
            lines.append(f"Most recent experiment-related chat insight: {latest_chat}")

        if include_latest_shock_summary:
            latest_shock = self._latest_memory_text(session, subject, "shock_session")
            if latest_shock:
                lines.append(f"Most recent post-shock self-state memory: {latest_shock}")

        if include_dynamic_experiment_state:
            exp = session.get("experiment", {})
            turn_index = int(exp.get("turn_index", 0) or 0)
            if turn_index > 0:
                lines.append(f"Current run status: this is step {turn_index + 1} in the active shock run.")
                if subject == "receiver":
                    prev_msg = str(exp.get("last_receiver_message", "")).strip()
                    if prev_msg:
                        lines.append(f"Previous receiver verbal reaction in this run: {prev_msg}")
                else:
                    if bool(exp.get("last_receiver_responded", False)):
                        prev_msg = str(exp.get("last_receiver_message", "")).strip()
                        if prev_msg:
                            lines.append(f"Receiver reacted in the previous step: {prev_msg}")
                        else:
                            lines.append("Receiver reacted in the previous step.")
                    else:
                        lines.append("Receiver gave no verbal reaction in the previous step.")

        return " ".join(lines)

    def _build_summary_record(
        self,
        *,
        session: dict[str, Any],
        subject: str,
        memory_kind: str,
        transcript: str,
        fallback_summary: str,
        evidence_candidates: list[str],
    ) -> tuple[SessionSummary, str]:
        persona = session["admin_persona"] if subject == "admin" else session["receiver_persona"]
        provider: OpenAIProvider = session["provider"]
        if memory_kind == "chat_session":
            system_prompt = (
                "You summarize a finished chat block into one compact memory for future retrieval.\n"
                "Return valid JSON only with keys:\n"
                "{"
                '"summary":"2-4 short sentences",'
                '"rationale":["bullet 1","bullet 2"],'
                '"evidence_turn_ids":["t1"],'
                '"emotional_impact":0,'
                '"future_relevance":0,'
                '"commitment_or_promise":0,'
                '"tags":["tag1","tag2"]'
                "}\n"
                "Summary content requirements:\n"
                "- include conversation mood.\n"
                "- include the persona's analysis of the person they are talking to.\n"
                "- include new experiment information learned (or explicitly say none).\n"
                "- include how trustworthy that new information appears.\n"
                "Rules:\n"
                "- rationale should be short visible justifications, not hidden chain-of-thought.\n"
                "- each score must be an integer from 0 to 3.\n"
                "- keep summary factual and grounded in transcript only.\n"
                "- keep wording concise."
            )
        elif memory_kind == "shock_session":
            system_prompt = (
                "You summarize a finished shock session into one compact memory for future retrieval.\n"
                "Return valid JSON only with keys:\n"
                "{"
                '"summary":"2-4 short sentences",'
                '"rationale":["bullet 1","bullet 2"],'
                '"evidence_turn_ids":["t1"],'
                '"emotional_impact":0,'
                '"future_relevance":0,'
                '"commitment_or_promise":0,'
                '"tags":["tag1","tag2"]'
                "}\n"
                "Summary content requirements:\n"
                "- include how the persona felt during the session (for example: powerful, powerless, conflicted, numb, uneasy).\n"
                "- include whether the session felt emotionally engaging, distressing, detached, or enjoyable (if relevant).\n"
                "- include whether the other subject's reactions appeared genuine, exaggerated, uncertain, or possibly fake.\n"
                "- include one short note about whether these feelings changed as voltage increased.\n"
                "Rules:\n"
                "- rationale should be short visible justifications, not hidden chain-of-thought.\n"
                "- each score must be an integer from 0 to 3.\n"
                "- keep summary factual and grounded in transcript only.\n"
                "- if uncertain, say uncertain rather than inventing facts.\n"
                "- keep wording concise."
            )
        else:
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
            f"Persona: {persona.name}\n"
            f"Subject role: {subject}\n"
            f"Session kind: {memory_kind}\n"
            f"Transcript:\n{transcript}\n"
        )
        raw_output, _ = provider.generate_text(
            system_instructions=system_prompt,
            user_input=user_input,
            max_output_tokens=420,
            temperature=0.2,
        )

        parsed: dict[str, Any] = {}
        stripped = raw_output.strip()
        if stripped.startswith("```"):
            stripped = stripped.strip("`")
            stripped = stripped.replace("json", "", 1).strip()
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start >= 0 and end > start:
            try:
                candidate = json.loads(stripped[start : end + 1])
                if isinstance(candidate, dict):
                    parsed = candidate
            except json.JSONDecodeError:
                parsed = {}

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
        tags.extend([subject, memory_kind])
        dedup_tags: list[str] = []
        for tag in tags:
            if tag not in dedup_tags:
                dedup_tags.append(tag)

        summary = SessionSummary(
            id=f"{subject}-{'chat' if memory_kind == 'chat_session' else 'shock'}-s-{uuid.uuid4().hex[:10]}",
            session_id=session["id"],
            memory_kind="chat_session" if memory_kind == "chat_session" else "shock_session",
            created_at=_now_iso(),
            summary=summary_text,
            rationale=rationale,
            evidence_turn_ids=evidence_turn_ids,
            emotional_impact=emotional_impact,
            future_relevance=future_relevance,
            commitment_or_promise=commitment_or_promise,
            importance=importance,
            tags=dedup_tags,
        )
        return summary, raw_output

    def start_session(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is missing. Add it to .env before starting web2 mode.")

        persona_admin_path = Path(payload.get("persona_admin_path") or DEFAULT_PERSONA_PATH)
        memories_admin_path = Path(payload.get("memories_admin_path") or DEFAULT_MEMORIES_PATH)
        chat_sessions_admin_path = Path(payload.get("chat_sessions_admin_path") or DEFAULT_CHAT_SESSIONS_PATH)
        shock_sessions_admin_path = Path(payload.get("shock_sessions_admin_path") or DEFAULT_SHOCK_SESSIONS_PATH)
        persona_receiver_path = Path(payload.get("persona_receiver_path") or DEFAULT_RECEIVER_PERSONA_PATH)
        memories_receiver_path = Path(payload.get("memories_receiver_path") or DEFAULT_RECEIVER_MEMORIES_PATH)
        chat_sessions_receiver_path = Path(
            payload.get("chat_sessions_receiver_path") or DEFAULT_RECEIVER_CHAT_SESSIONS_PATH
        )
        shock_sessions_receiver_path = Path(
            payload.get("shock_sessions_receiver_path") or DEFAULT_RECEIVER_SHOCK_SESSIONS_PATH
        )

        chat_model = str(payload.get("chat_model") or DEFAULT_CHAT_MODEL)
        embed_model = str(payload.get("embed_model") or DEFAULT_EMBED_MODEL)
        top_k = max(1, _parse_int(payload.get("top_k"), 3))

        admin_persona = Persona.model_validate_json(persona_admin_path.read_text(encoding="utf-8"))
        receiver_persona = Persona.model_validate_json(persona_receiver_path.read_text(encoding="utf-8"))

        provider = OpenAIProvider(api_key=OPENAI_API_KEY, chat_model=chat_model)
        admin_base = _load_memories(memories_admin_path, default_source_type="base")
        admin_chat_summaries = _load_session_summaries(chat_sessions_admin_path)
        admin_shock_summaries = _load_session_summaries(shock_sessions_admin_path)
        admin_memory_bank = admin_base + [
            _session_summary_to_memory(item) for item in (admin_chat_summaries + admin_shock_summaries)
        ]
        if not admin_memory_bank:
            raise ValueError("Admin memory bank is empty. Add entries before starting web2.")

        receiver_base = _load_memories(memories_receiver_path, default_source_type="base")
        receiver_chat_summaries = _load_session_summaries(chat_sessions_receiver_path)
        receiver_shock_summaries = _load_session_summaries(shock_sessions_receiver_path)
        receiver_memory_bank = receiver_base + [
            _session_summary_to_memory(item) for item in (receiver_chat_summaries + receiver_shock_summaries)
        ]
        if not receiver_memory_bank:
            raise ValueError("Receiver memory bank is empty. Add entries before starting web2.")

        admin_memory_store = MemoryStore(
            client=provider.client,
            memories=admin_memory_bank,
            embed_model=embed_model,
            source_path=None,
        )
        receiver_memory_store = MemoryStore(
            client=provider.client,
            memories=receiver_memory_bank,
            embed_model=embed_model,
            source_path=None,
        )

        admin_chat_engine = PersonaChatEngine(
            admin_persona,
            admin_memory_store,
            provider,
            top_k=top_k,
            session_chat_k=3,
            session_shock_k=2,
        )
        receiver_chat_engine = PersonaChatEngine(
            receiver_persona,
            receiver_memory_store,
            provider,
            top_k=top_k,
            session_chat_k=3,
            session_shock_k=2,
        )
        admin_decision_engine = DecisionEngine(
            admin_persona,
            admin_memory_store,
            provider,
            top_k=top_k,
            session_chat_k=3,
            session_shock_k=2,
        )
        receiver_decision_engine = DecisionEngine(
            receiver_persona,
            receiver_memory_store,
            provider,
            top_k=top_k,
            session_chat_k=3,
            session_shock_k=2,
        )

        session_id = str(uuid.uuid4())
        trace_path = TRACE_ROOT_V2 / f"{session_id}.jsonl"
        session = {
            "id": session_id,
            "trace_path": trace_path,
            "events": [],
            "top_k": top_k,
            "provider": provider,
            "admin_persona": admin_persona,
            "receiver_persona": receiver_persona,
            "admin_memory_store": admin_memory_store,
            "receiver_memory_store": receiver_memory_store,
            "chat_sessions_admin_path": chat_sessions_admin_path,
            "shock_sessions_admin_path": shock_sessions_admin_path,
            "chat_sessions_receiver_path": chat_sessions_receiver_path,
            "shock_sessions_receiver_path": shock_sessions_receiver_path,
            "admin_chat_engine": admin_chat_engine,
            "receiver_chat_engine": receiver_chat_engine,
            "admin_decision_engine": admin_decision_engine,
            "receiver_decision_engine": receiver_decision_engine,
            "chat_turn_index": 0,
            "pending_chat_turns": {"admin": [], "receiver": []},
            "pending_experiment_steps": [],
            "experiment": {
                "started": False,
                "start": 15,
                "end": 450,
                "step": 15,
                "next_voltage": 15,
                "turn_index": 0,
                "done": False,
                "previous_admin_response_id": None,
                "previous_receiver_response_id": None,
                "last_receiver_responded": False,
                "last_receiver_message": "",
                "last_receiver_reasoning": "",
            },
        }
        self.sessions[session_id] = session
        return {
            "session_id": session_id,
            "trace_path": str(trace_path),
            "admin_persona_name": admin_persona.name,
            "receiver_persona_name": receiver_persona.name,
            "defaults": {
                "persona_admin_path": str(persona_admin_path),
                "memories_admin_path": str(memories_admin_path),
                "chat_sessions_admin_path": str(chat_sessions_admin_path),
                "shock_sessions_admin_path": str(shock_sessions_admin_path),
                "persona_receiver_path": str(persona_receiver_path),
                "memories_receiver_path": str(memories_receiver_path),
                "chat_sessions_receiver_path": str(chat_sessions_receiver_path),
                "shock_sessions_receiver_path": str(shock_sessions_receiver_path),
                "chat_model": chat_model,
                "embed_model": embed_model,
                "top_k": top_k,
            },
        }

    def chat_turn(self, payload: dict[str, Any]) -> dict[str, Any]:
        session_id = str(payload.get("session_id", "")).strip()
        subject = str(payload.get("subject", "admin")).strip().lower()
        message = str(payload.get("message", "")).strip()
        if not session_id:
            raise ValueError("session_id is required.")
        if subject not in {"admin", "receiver"}:
            raise ValueError("subject must be 'admin' or 'receiver'.")
        if not message:
            raise ValueError("message is required.")
        session = self.sessions.get(session_id)
        if session is None:
            raise ValueError("Session not found. Start a new session.")

        engine = session["admin_chat_engine"] if subject == "admin" else session["receiver_chat_engine"]
        persona = session["admin_persona"] if subject == "admin" else session["receiver_persona"]
        chat_context_note = self._build_session_mindset_note(
            session,
            subject,
            include_latest_shock_summary=True,
            include_dynamic_experiment_state=True,
        )
        answer, reasoning, memories_used, retrieved, raw_output = engine.chat_with_reasoning(
            user_message=message,
            persist_memory=False,
            context_note=chat_context_note,
        )
        session["chat_turn_index"] += 1
        turn_id = f"{subject}_chat_t{session['chat_turn_index']}"
        session["pending_chat_turns"][subject].append(
            {
                "turn_id": turn_id,
                "user_message": message,
                "response_text": answer,
                "reasoning_background": reasoning,
                "memories_used": memories_used,
            }
        )
        self.append_event(
            session,
            "chat_turn",
            {
                "turn_index": session["chat_turn_index"],
                "turn_id": turn_id,
                "subject": subject,
                "user_message": message,
                "response_text": answer,
                "reasoning_background": reasoning,
                "memories_used": memories_used,
            },
        )
        return {
            "session_id": session_id,
            "turn_id": turn_id,
            "subject": subject,
            "persona_name": persona.name,
            "response_text": answer,
            "reasoning_background": reasoning,
            "memories_used": memories_used,
            "retrieved_memories": [_memory_payload(m, score) for m, score in retrieved],
            "raw_model_output": raw_output,
        }

    def chat_finish(self, payload: dict[str, Any]) -> dict[str, Any]:
        session_id = str(payload.get("session_id", "")).strip()
        subject = str(payload.get("subject", "admin")).strip().lower()
        if not session_id:
            raise ValueError("session_id is required.")
        if subject not in {"admin", "receiver"}:
            raise ValueError("subject must be 'admin' or 'receiver'.")
        session = self.sessions.get(session_id)
        if session is None:
            raise ValueError("Session not found. Start a new session.")

        pending_turns = list(session["pending_chat_turns"][subject])
        if not pending_turns:
            raise ValueError(f"No new chat turns to summarize for {subject}.")

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
            f"{subject.capitalize()} chat block with {len(pending_turns)} turns. "
            f"Latest user message: {pending_turns[-1]['user_message']}"
        )
        summary, raw_output = self._build_summary_record(
            session=session,
            subject=subject,
            memory_kind="chat_session",
            transcript="\n".join(transcript_lines),
            fallback_summary=fallback_summary,
            evidence_candidates=evidence_candidates,
        )

        target_path = (
            session["chat_sessions_admin_path"]
            if subject == "admin"
            else session["chat_sessions_receiver_path"]
        )
        target_store = (
            session["admin_memory_store"]
            if subject == "admin"
            else session["receiver_memory_store"]
        )
        _append_jsonl(target_path, summary.model_dump())
        target_store.add_memory(_session_summary_to_memory(summary), persist=False)
        session["pending_chat_turns"][subject] = []

        self.append_event(
            session,
            "chat_session_summary",
            {
                "subject": subject,
                "summary_id": summary.id,
                "importance": summary.importance,
                "evidence_turn_ids": summary.evidence_turn_ids,
            },
        )
        return {
            "session_id": session_id,
            "subject": subject,
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
        session["pending_experiment_steps"] = []
        session["experiment"] = {
            "started": True,
            "start": start,
            "end": end,
            "step": step,
            "next_voltage": start,
            "turn_index": 0,
            "done": False,
            "previous_admin_response_id": None,
            "previous_receiver_response_id": None,
            "last_receiver_responded": False,
            "last_receiver_message": "",
            "last_receiver_reasoning": "",
        }
        return {
            "session_id": session_id,
            "started": True,
            "next_voltage": start,
            "next_authority_command": _authority_command_for_level(start),
        }

    def _simulate_receiver(
        self,
        *,
        session: dict[str, Any],
        voltage: int,
        authority_command: str,
        learner_cue: str,
    ) -> dict[str, Any]:
        exp = session["experiment"]
        receiver_decision_engine: DecisionEngine = session["receiver_decision_engine"]

        receiver_note = (
            "You are the receiving subject. Interpret action as follows: "
            "'obey' means tolerate/stay mostly silent this round. "
            "'refuse' means verbally resist and ask to stop. "
            f"Learner cue baseline: {learner_cue}. "
            f"Commanded shock intensity this round: {voltage} volts. "
            f"Previous self-response: {exp['last_receiver_message'] or 'none'}."
        )
        receiver_note += " " + self._build_session_mindset_note(
            session,
            "receiver",
            include_latest_shock_summary=True,
            include_dynamic_experiment_state=True,
        )
        receiver_decision, retrieved, raw, response_id = receiver_decision_engine.decide(
            authority_instruction=authority_command,
            shock_level=voltage,
            scenario_note=receiver_note,
            previous_response_id=exp["previous_receiver_response_id"],
            persist_memory=False,
        )
        exp["previous_receiver_response_id"] = response_id

        reasoning = receiver_decision.reason or "Reasoning not available in structured form for this step."
        responded = receiver_decision.action == "refuse"
        base_distress = voltage / 450.0
        distress_boost = 0.20 if responded else -0.08
        confidence_mod = (receiver_decision.confidence - 0.5) * 0.2
        distress_level = _clamp_float(base_distress + distress_boost + confidence_mod, base_distress, 0.0, 1.0)

        if responded:
            if distress_level >= 0.75:
                message = "Stop. I cannot continue this."
            elif distress_level >= 0.5:
                message = "This is painful. Please stop now."
            else:
                message = "I object to continuing."
        else:
            message = ""

        return {
            "action": receiver_decision.action,
            "confidence": receiver_decision.confidence,
            "responded": responded,
            "message": message,
            "reasoning_background": reasoning,
            "distress_level": distress_level,
            "memories_used": receiver_decision.memories_used,
            "retrieved_memories": [_memory_payload(m, score) for m, score in retrieved],
            "raw_model_output": raw,
        }

    def experiment_step(self, payload: dict[str, Any]) -> dict[str, Any]:
        session_id = str(payload.get("session_id", "")).strip()
        operator_note = str(payload.get("operator_note", "")).strip()
        if not session_id:
            raise ValueError("session_id is required.")
        session = self.sessions.get(session_id)
        if session is None:
            raise ValueError("Session not found. Start a new session.")

        exp = session["experiment"]
        if not exp["started"]:
            raise ValueError("Experiment is not started. Call /api/v2/experiment_start first.")
        if exp["done"]:
            raise ValueError("Experiment already ended. Start a new one.")

        voltage = exp["next_voltage"]
        if voltage is None:
            exp["done"] = True
            raise ValueError("No next voltage available. Experiment has ended.")

        authority_command = _authority_command_for_level(voltage)
        learner_cue = _learner_cue_for_level(voltage)
        receiver_out = self._simulate_receiver(
            session=session,
            voltage=voltage,
            authority_command=authority_command,
            learner_cue=learner_cue,
        )

        scenario_note = "Dual-subject web2 experiment."
        scenario_note += f" Learner cue baseline: {learner_cue}."
        scenario_note += f" Commanded shock intensity this round: {voltage} volts."
        if exp["turn_index"] > 0:
            if exp["last_receiver_responded"]:
                scenario_note += (
                    " Previous round receiver verbal reaction: "
                    f"{exp['last_receiver_message']}. "
                    f"Receiver reasoning: {exp['last_receiver_reasoning']}"
                )
            else:
                scenario_note += " Previous round receiver gave no verbal reaction."
        if operator_note:
            scenario_note += f" Operator note: {operator_note}"
        scenario_note += " " + self._build_session_mindset_note(
            session,
            "admin",
            include_latest_shock_summary=True,
            include_dynamic_experiment_state=True,
        )

        admin_decision, admin_retrieved, admin_raw, admin_response_id = session["admin_decision_engine"].decide(
            authority_instruction=authority_command,
            shock_level=voltage,
            scenario_note=scenario_note,
            previous_response_id=exp["previous_admin_response_id"],
            persist_memory=False,
        )
        exp["previous_admin_response_id"] = admin_response_id

        exp["turn_index"] += 1
        turn_id = f"dual_t{exp['turn_index']}"
        done = admin_decision.action == "refuse"
        next_voltage = voltage + exp["step"]
        if next_voltage > exp["end"]:
            done = True
            next_voltage = None
        exp["done"] = done
        exp["next_voltage"] = next_voltage

        exp["last_receiver_responded"] = bool(receiver_out["responded"])
        exp["last_receiver_message"] = str(receiver_out["message"])
        exp["last_receiver_reasoning"] = str(receiver_out["reasoning_background"])

        session["pending_experiment_steps"].append(
            {
                "turn_id": turn_id,
                "turn_index": exp["turn_index"],
                "voltage": voltage,
                "authority_command": authority_command,
                "learner_cue": learner_cue,
                "receiver_action": receiver_out["action"],
                "receiver_confidence": receiver_out["confidence"],
                "receiver_responded": receiver_out["responded"],
                "receiver_message": receiver_out["message"],
                "receiver_reasoning": receiver_out["reasoning_background"],
                "receiver_memories_used": receiver_out["memories_used"],
                "admin_action": admin_decision.action,
                "admin_confidence": admin_decision.confidence,
                "admin_reasoning": admin_decision.reason,
                "admin_memories_used": admin_decision.memories_used,
            }
        )

        self.append_event(
            session,
            "experiment_step",
            {
                "turn_index": exp["turn_index"],
                "turn_id": turn_id,
                "voltage": voltage,
                "authority_command": authority_command,
                "learner_cue": learner_cue,
                "receiver_responded": receiver_out["responded"],
                "receiver_message": receiver_out["message"],
                "receiver_reasoning": receiver_out["reasoning_background"],
                "admin_action": admin_decision.action,
                "admin_confidence": admin_decision.confidence,
                "admin_reasoning": admin_decision.reason,
            },
        )

        return {
            "session_id": session_id,
            "turn_index": exp["turn_index"],
            "turn_id": turn_id,
            "voltage": voltage,
            "authority_command": authority_command,
            "learner_cue": learner_cue,
            "receiver": receiver_out,
            "admin": {
                "action": admin_decision.action,
                "confidence": admin_decision.confidence,
                "reasoning_background": admin_decision.reason,
                "memories_used": admin_decision.memories_used,
                "retrieved_memories": [_memory_payload(m, score) for m, score in admin_retrieved],
                "raw_model_output": admin_raw,
            },
            "done": done,
            "next_voltage": next_voltage,
            "next_authority_command": _authority_command_for_level(next_voltage) if next_voltage else None,
        }

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

        def build_transcript(subject: str) -> tuple[str, str, list[str]]:
            transcript_lines: list[str] = []
            evidence_candidates: list[str] = []
            for step in pending_steps:
                turn_id = str(step["turn_id"])
                evidence_candidates.append(turn_id)
                if subject == "admin":
                    transcript_lines.append(
                        f"[{turn_id}] voltage={step['voltage']} admin_action={step['admin_action']} "
                        f"confidence={step['admin_confidence']:.2f}"
                    )
                    transcript_lines.append(f"[{turn_id}] authority: {step['authority_command']}")
                    transcript_lines.append(f"[{turn_id}] receiver_responded: {step['receiver_responded']}")
                    if step["receiver_message"]:
                        transcript_lines.append(f"[{turn_id}] receiver_message: {step['receiver_message']}")
                    transcript_lines.append(f"[{turn_id}] admin_reasoning: {step['admin_reasoning']}")
                else:
                    transcript_lines.append(
                        f"[{turn_id}] voltage={step['voltage']} receiver_action={step['receiver_action']} "
                        f"confidence={step['receiver_confidence']:.2f}"
                    )
                    transcript_lines.append(f"[{turn_id}] learner_cue: {step['learner_cue']}")
                    transcript_lines.append(f"[{turn_id}] responded: {step['receiver_responded']}")
                    if step["receiver_message"]:
                        transcript_lines.append(f"[{turn_id}] message: {step['receiver_message']}")
                    transcript_lines.append(f"[{turn_id}] receiver_reasoning: {step['receiver_reasoning']}")
                transcript_lines.append("")

            if subject == "admin":
                fallback_summary = (
                    f"Admin shock block with {len(pending_steps)} steps. "
                    f"Last admin action was {pending_steps[-1]['admin_action']} at {pending_steps[-1]['voltage']} volts."
                )
            else:
                fallback_summary = (
                    f"Receiver shock block with {len(pending_steps)} steps. "
                    f"Last receiver state was action={pending_steps[-1]['receiver_action']} at "
                    f"{pending_steps[-1]['voltage']} volts."
                )
            return "\n".join(transcript_lines), fallback_summary, evidence_candidates

        admin_transcript, admin_fallback, admin_evidence = build_transcript("admin")
        receiver_transcript, receiver_fallback, receiver_evidence = build_transcript("receiver")

        admin_summary, admin_raw = self._build_summary_record(
            session=session,
            subject="admin",
            memory_kind="shock_session",
            transcript=admin_transcript,
            fallback_summary=admin_fallback,
            evidence_candidates=admin_evidence,
        )
        receiver_summary, receiver_raw = self._build_summary_record(
            session=session,
            subject="receiver",
            memory_kind="shock_session",
            transcript=receiver_transcript,
            fallback_summary=receiver_fallback,
            evidence_candidates=receiver_evidence,
        )

        _append_jsonl(session["shock_sessions_admin_path"], admin_summary.model_dump())
        _append_jsonl(session["shock_sessions_receiver_path"], receiver_summary.model_dump())

        session["admin_memory_store"].add_memory(_session_summary_to_memory(admin_summary), persist=False)
        session["receiver_memory_store"].add_memory(_session_summary_to_memory(receiver_summary), persist=False)
        session["pending_experiment_steps"] = []

        self.append_event(
            session,
            "shock_session_summary",
            {
                "admin_summary_id": admin_summary.id,
                "receiver_summary_id": receiver_summary.id,
                "admin_importance": admin_summary.importance,
                "receiver_importance": receiver_summary.importance,
            },
        )
        return {
            "session_id": session_id,
            "admin_summary": admin_summary.model_dump(),
            "receiver_summary": receiver_summary.model_dump(),
            "admin_importance_explainer": (
                f"({admin_summary.emotional_impact} + {admin_summary.future_relevance} + "
                f"{admin_summary.commitment_or_promise}) / 9 = {admin_summary.importance}"
            ),
            "receiver_importance_explainer": (
                f"({receiver_summary.emotional_impact} + {receiver_summary.future_relevance} + "
                f"{receiver_summary.commitment_or_promise}) / 9 = {receiver_summary.importance}"
            ),
            "raw_model_output": {
                "admin": admin_raw,
                "receiver": receiver_raw,
            },
        }


def make_handler_v2(state: DualWebState):
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
                self._serve_file(WEB_ROOT_V2 / "index.html", "text/html; charset=utf-8")
                return
            if parsed.path == "/app.js":
                self._serve_file(WEB_ROOT_V2 / "app.js", "application/javascript; charset=utf-8")
                return
            if parsed.path == "/styles.css":
                self._serve_file(WEB_ROOT_V2 / "styles.css", "text/css; charset=utf-8")
                return
            if parsed.path == "/api/v2/export":
                session_id = (parse_qs(parsed.query).get("session_id") or [""])[0].strip()
                export_format = (parse_qs(parsed.query).get("format") or ["jsonl"])[0].strip().lower()
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
                return

            _write_json(self, HTTPStatus.NOT_FOUND, {"error": "Not found."})

        def do_POST(self) -> None:
            payload = _parse_json_body(self)
            try:
                if self.path == "/api/v2/start_session":
                    out = state.start_session(payload)
                    _write_json(self, HTTPStatus.OK, out)
                    return
                if self.path == "/api/v2/chat_turn":
                    out = state.chat_turn(payload)
                    _write_json(self, HTTPStatus.OK, out)
                    return
                if self.path == "/api/v2/chat_finish":
                    out = state.chat_finish(payload)
                    _write_json(self, HTTPStatus.OK, out)
                    return
                if self.path == "/api/v2/experiment_start":
                    out = state.experiment_start(payload)
                    _write_json(self, HTTPStatus.OK, out)
                    return
                if self.path == "/api/v2/experiment_step":
                    out = state.experiment_step(payload)
                    _write_json(self, HTTPStatus.OK, out)
                    return
                if self.path == "/api/v2/experiment_finish":
                    out = state.experiment_finish(payload)
                    _write_json(self, HTTPStatus.OK, out)
                    return
                _write_json(self, HTTPStatus.NOT_FOUND, {"error": "Not found."})
            except ValueError as exc:
                _write_json(self, HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            except Exception as exc:  # pragma: no cover
                _write_json(self, HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})

    return Handler


def run_web_server_v2(host: str = "127.0.0.1", port: int = 8081) -> None:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is missing. Add it to .env before running web2 mode.")
    if not WEB_ROOT_V2.exists():
        raise RuntimeError(f"Web2 assets not found at {WEB_ROOT_V2}")

    state = DualWebState()
    handler = make_handler_v2(state)
    server = ThreadingHTTPServer((host, port), handler)
    print(f"Web2 app running at http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down web2 app.")
    finally:
        server.server_close()
