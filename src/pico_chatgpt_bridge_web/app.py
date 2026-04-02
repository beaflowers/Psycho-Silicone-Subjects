"""Flask app for a mockumentary-style chat interface."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import random
from pathlib import Path

from flask import Flask, jsonify, render_template, request

from src.pico_chatgpt_bridge.openai_client import create_client, get_model_name
from src.pico_chatgpt_bridge.prompting import (
    build_live_instruction,
    build_shift_instruction,
    choose_random_memory,
    choose_random_mood,
    clamp_shift,
)
from src.pico_chatgpt_bridge.rag_engine import RAGEngine

from .store import ChatStateStore


APP_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = APP_ROOT.parents[1]
STORE_PATH = PROJECT_ROOT / "data" / "web_chat_state.json"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _extract_response_text(response: object) -> str:
    """Return readable text from a Responses API result."""
    direct_text = getattr(response, "output_text", "") or ""
    if direct_text.strip():
        return direct_text.strip()

    output_items = getattr(response, "output", None) or []
    text_parts: list[str] = []

    for item in output_items:
        content_items = getattr(item, "content", None) or []
        for content in content_items:
            if getattr(content, "type", "") == "output_text":
                text_value = getattr(content, "text", "") or ""
                if text_value.strip():
                    text_parts.append(text_value.strip())
    return "\n\n".join(text_parts)


def _serialise_chunk(chunk: object) -> dict[str, object]:
    return {
        "text": getattr(chunk, "text", ""),
        "score": getattr(chunk, "score", 0.0),
        "source_path": getattr(chunk, "source_path", ""),
        "persona": getattr(chunk, "persona", ""),
    }


def _new_chat_state(
    *,
    conversation_id: str,
    title: str = "Subject interview log",
    shift: float = 0.0,
) -> dict[str, object]:
    now = _utc_now()
    return {
        "conversation_id": conversation_id,
        "title": title,
        "shift": shift,
        "mood": None,
        "active_memories": [],
        "messages": [],
        "created_at": now,
        "updated_at": now,
    }


def create_app() -> Flask:
    app = Flask(
        __name__,
        template_folder=str(APP_ROOT / "templates"),
        static_folder=str(APP_ROOT / "static"),
    )
    app.config["JSON_SORT_KEYS"] = False

    client = create_client()
    model = get_model_name()
    rng = random.Random()
    store = ChatStateStore(STORE_PATH)

    try:
        rag = RAGEngine(client=client, chat_model=model)
        rag.ensure_ready()
        rag_error: str | None = None
    except Exception as exc:  # pragma: no cover - defensive startup fallback
        rag = None
        rag_error = str(exc)

    @app.get("/")
    def index() -> str:
        return render_template(
            "index.html",
            rag_sources=rag.describe_sources() if rag is not None else "RAG unavailable",
            rag_error=rag_error,
            model_name=model,
        )

    @app.get("/api/chat/state")
    def get_state() -> object:
        state = store.load()
        if state is None:
            conversation = client.conversations.create(
                metadata={"surface": "mockumentary", "mode": "single-state"}
            )
            state = _new_chat_state(conversation_id=conversation.id)
            store.save(state)
        return jsonify(state)

    @app.post("/api/chat/reset")
    def reset_state() -> tuple[object, int]:
        payload = request.get_json(silent=True) or {}
        title = str(payload.get("title") or "Subject interview log").strip()
        shift = clamp_shift(payload.get("shift", 0.0))
        conversation = client.conversations.create(
            metadata={"surface": "mockumentary", "mode": "single-state"}
        )
        state = _new_chat_state(
            conversation_id=conversation.id,
            title=title,
            shift=shift,
        )
        store.save(state)
        return jsonify(state), 201

    @app.patch("/api/chat/state")
    def update_state() -> object:
        state = store.load()
        if state is None:
            return jsonify({"error": "Chat state not found."}), 404

        payload = request.get_json(silent=True) or {}
        updates: dict[str, object] = {}

        if "shift" in payload:
            updates["shift"] = clamp_shift(payload["shift"])
        if "mood" in payload:
            mood = payload["mood"]
            updates["mood"] = str(mood).strip() if mood else None
        if "active_memories" in payload and isinstance(payload["active_memories"], list):
            updates["active_memories"] = [
                str(item).strip() for item in payload["active_memories"] if str(item).strip()
            ][-5:]

        updates["updated_at"] = _utc_now()
        state.update(updates)
        updated = store.save(state)
        return jsonify(updated)

    @app.post("/api/chat/mood")
    def randomise_mood() -> object:
        state = store.load()
        if state is None:
            return jsonify({"error": "Chat state not found."}), 404

        mood = choose_random_mood(rng)
        state.update({"mood": mood, "updated_at": _utc_now()})
        updated = store.save(state)
        return jsonify({"mood": mood, "state": updated})

    @app.post("/api/chat/memory")
    def add_memory() -> object:
        state = store.load()
        if state is None:
            return jsonify({"error": "Chat state not found."}), 404

        memory = choose_random_memory(state.get("active_memories", []), rng)
        if not memory:
            return jsonify({"error": "No memory files were found in data/."}), 400

        active_memories = list(state.get("active_memories", []))
        active_memories.append(memory)
        active_memories = active_memories[-5:]
        state.update({"active_memories": active_memories, "updated_at": _utc_now()})
        updated = store.save(state)
        return jsonify({"memory": memory, "state": updated})

    @app.post("/api/chat/message")
    def create_message() -> object:
        state = store.load()
        if state is None:
            return jsonify({"error": "Chat state not found."}), 404

        payload = request.get_json(silent=True) or {}
        user_text = str(payload.get("message") or "").strip()
        if not user_text:
            return jsonify({"error": "Message is required."}), 400

        shift = clamp_shift(state.get("shift", 0.0))
        mood = state.get("mood")
        active_memories = list(state.get("active_memories", []))
        live_instruction = build_live_instruction(mood, active_memories)

        retrieved_chunks: list[dict[str, object]] = []
        context_block = "No retrieved context was available for this turn."
        retrieval_note = rag_error
        if rag is not None:
            chunks = rag.retrieve(user_text, shift=shift)
            retrieved_chunks = [_serialise_chunk(chunk) for chunk in chunks]
            context_block = "\n\n---\n\n".join(
                (
                    f"[Chunk {index + 1} | persona {chunk['persona']} | score {chunk['score']:.3f}]\n"
                    f"Source: {chunk['source_path']}\n"
                    f"{chunk['text']}"
                )
                for index, chunk in enumerate(retrieved_chunks)
            )
            retrieval_note = None

        instructions = build_shift_instruction(shift)
        input_parts = [
            f"Context:\n{context_block}",
            f"Question: {user_text}",
        ]
        if live_instruction:
            input_parts.append(live_instruction)
        if retrieval_note:
            input_parts.append(f"RAG note: {retrieval_note}")

        response = client.responses.create(
            model=model,
            conversation=state["conversation_id"],
            instructions=instructions,
            input="\n\n".join(input_parts),
        )
        assistant_text = _extract_response_text(response) or "[No text output returned.]"

        messages = list(state.get("messages", []))
        user_message = {
            "role": "user",
            "content": user_text,
            "created_at": _utc_now(),
        }
        assistant_message = {
            "role": "assistant",
            "content": assistant_text,
            "response_id": response.id,
            "retrieved_context": retrieved_chunks,
            "created_at": _utc_now(),
        }
        messages.extend([user_message, assistant_message])

        state.update(
            {
                "messages": messages,
                "updated_at": _utc_now(),
            }
        )
        updated = store.save(state)
        return jsonify(
            {
                "state": updated,
                "assistant": assistant_message,
                "conversation_id": state["conversation_id"],
            }
        )

    @app.get("/api/chat/export")
    def export_state() -> object:
        state = store.load()
        if state is None:
            return jsonify({"error": "Chat state not found."}), 404
        return app.response_class(
            response=json.dumps(state, indent=2, ensure_ascii=True),
            mimetype="application/json",
        )

    return app


app = create_app()


if __name__ == "__main__":
    app.run(debug=True)
