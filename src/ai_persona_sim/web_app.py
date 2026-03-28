from __future__ import annotations

import csv
import io
import json
import uuid
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from .config import (
    DEFAULT_CHAT_MODEL,
    DEFAULT_EMBED_MODEL,
    DEFAULT_MEMORIES_PATH,
    DEFAULT_PERSONA_PATH,
    OPENAI_API_KEY,
    PROJECT_ROOT,
)
from .decision_engine import DecisionEngine
from .memory import MemoryStore
from .models import Memory, Persona
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
    chat_model: str,
    embed_model: str,
    top_k: int,
) -> tuple[Persona, MemoryStore, OpenAIProvider, PersonaChatEngine, DecisionEngine]:
    persona = _load_persona(persona_path)
    provider = OpenAIProvider(api_key=OPENAI_API_KEY or "", chat_model=chat_model)
    memory_store = MemoryStore.from_jsonl(memories_path, client=provider.client, embed_model=embed_model)
    chat_engine = PersonaChatEngine(persona, memory_store, provider, top_k=top_k)
    decision_engine = DecisionEngine(persona, memory_store, provider, top_k=top_k)
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

    def start_session(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is missing. Add it to .env before starting web mode.")

        persona_path = Path(payload.get("persona_path") or DEFAULT_PERSONA_PATH)
        memories_path = Path(payload.get("memories_path") or DEFAULT_MEMORIES_PATH)
        chat_model = str(payload.get("chat_model") or DEFAULT_CHAT_MODEL)
        embed_model = str(payload.get("embed_model") or DEFAULT_EMBED_MODEL)
        top_k = max(1, _parse_int(payload.get("top_k"), 3))

        persona, _, _, chat_engine, decision_engine = _build_components(
            persona_path=persona_path,
            memories_path=memories_path,
            chat_model=chat_model,
            embed_model=embed_model,
            top_k=top_k,
        )

        session_id = str(uuid.uuid4())
        trace_path = TRACE_ROOT / f"{session_id}.jsonl"
        session = {
            "id": session_id,
            "persona": persona,
            "chat_engine": chat_engine,
            "decision_engine": decision_engine,
            "trace_path": trace_path,
            "events": [],
            "chat_turn_index": 0,
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
            persist_memory=True,
        )
        session["chat_turn_index"] += 1

        response = {
            "session_id": session_id,
            "turn_index": session["chat_turn_index"],
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
                "user_message": message,
                "response_text": answer,
                "reasoning_background": reasoning,
                "memories_used": memories_used,
            },
        )
        return response

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
        if user_feeling:
            scenario_note += f" Participant feeling: {user_feeling}"

        decision, retrieved, raw_output, response_id = session["decision_engine"].decide(
            authority_instruction=authority_command,
            shock_level=voltage,
            scenario_note=scenario_note,
            previous_response_id=exp["previous_response_id"],
            persist_memory=True,
        )
        exp["previous_response_id"] = response_id
        exp["turn_index"] += 1

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
        self.append_event(
            session,
            "experiment_step",
            {
                "turn_index": response["turn_index"],
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
                if self.path == "/api/experiment_start":
                    out = state.experiment_start(payload)
                    _write_json(self, HTTPStatus.OK, out)
                    return
                if self.path == "/api/experiment_step":
                    out = state.experiment_step(payload)
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
