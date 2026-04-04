"""Flask app for the two-character Psycho-Silicone web chat."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import random
from pathlib import Path
from typing import Any
from urllib.parse import quote

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
PDF_ROOT = PROJECT_ROOT / "PDFs"
STATIC_MEDIA_ROOT = APP_ROOT / "static" / "character_media"

SILICONE_KEY = "silicone_subject"
JEKYLL_KEY = "jekyll_hyde"
DEFAULT_ACTIVE_CHARACTER = SILICONE_KEY

CHARACTER_CONFIGS = {
    SILICONE_KEY: {
        "label": "Angela / Housewife",
        "title": "Silicone Subject interview log",
        "empty_message": "Ask Angela/Housewife anything from the archive.",
        "default_shift": 0.9,
        "subtitle": "Composite archive with a continuous shift between Angela Carter and the housewife voice.",
    },
    JEKYLL_KEY: {
        "label": "Jekyll / Hyde",
        "title": "Jekyll / Hyde interview log",
        "empty_message": "Ask Jekyll/Hyde anything from the case notes.",
        "default_shift": 0.0,
        "subtitle": "Separate archive built from the Jekyll and Hyde corpus.",
    },
}

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".webm", ".mov", ".m4v"}

JEKYLL_HYDE_INSTRUCTIONS = (
    "You are one unstable self shaped by Dr. Jekyll and Mr. Hyde. "
    "The retrieved text is not outside reference material; it is fragmented memory, "
    "confession, repression, and self-justification. Let either Jekyll or Hyde dominate "
    "the reply based on the emotional pressure of the moment, but do not split into two "
    "separate speakers in a single answer. Preserve continuity with the prior conversation "
    "state, stay grounded in the retrieved text, and answer in 2-4 vivid sentences."
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _extract_response_text(response: object) -> str:
    direct_text = getattr(response, "output_text", "") or ""
    if direct_text.strip():
        return direct_text.strip()

    output_items = getattr(response, "output", None) or []
    text_parts: list[str] = []
    for item in output_items:
        content_items = getattr(item, "content", None) or []
        for content in content_items:
            if getattr(content, "type", "") != "output_text":
                continue
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


def _new_character_state(
    character_key: str,
    *,
    conversation_id: str = "",
    title: str | None = None,
    shift: float | None = None,
) -> dict[str, object]:
    config = CHARACTER_CONFIGS[character_key]
    now = _utc_now()
    default_shift = config["default_shift"] if shift is None else clamp_shift(shift)
    return {
        "character_key": character_key,
        "title": title or config["title"],
        "conversation_id": conversation_id,
        "shift": default_shift,
        "mood": None,
        "active_memories": [],
        "messages": [],
        "created_at": now,
        "updated_at": now,
    }


def _ensure_messages(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, dict)]


def _normalise_character_state(character_key: str, payload: object) -> dict[str, object]:
    if not isinstance(payload, dict):
        return _new_character_state(character_key)

    base = _new_character_state(
        character_key,
        conversation_id=str(payload.get("conversation_id", "") or ""),
        title=str(payload.get("title", "") or CHARACTER_CONFIGS[character_key]["title"]),
        shift=payload.get("shift"),
    )
    base["mood"] = payload.get("mood")
    base["active_memories"] = [
        item for item in payload.get("active_memories", []) if isinstance(item, str)
    ]
    base["messages"] = _ensure_messages(payload.get("messages"))
    base["created_at"] = str(payload.get("created_at", base["created_at"]))
    base["updated_at"] = str(payload.get("updated_at", base["updated_at"]))
    return base


def _migrate_legacy_state(raw_state: dict[str, Any] | None) -> dict[str, Any]:
    now = _utc_now()
    default_state = {
        "active_character": DEFAULT_ACTIVE_CHARACTER,
        "characters": {
            SILICONE_KEY: _new_character_state(SILICONE_KEY),
            JEKYLL_KEY: _new_character_state(JEKYLL_KEY),
        },
        "updated_at": now,
    }

    if not raw_state:
        return default_state

    if isinstance(raw_state.get("characters"), dict):
        characters = dict(raw_state["characters"])
        default_state["active_character"] = str(
            raw_state.get("active_character", DEFAULT_ACTIVE_CHARACTER)
        )
        default_state["characters"][SILICONE_KEY] = _normalise_character_state(
            SILICONE_KEY,
            characters.get(SILICONE_KEY),
        )
        default_state["characters"][JEKYLL_KEY] = _normalise_character_state(
            JEKYLL_KEY,
            characters.get(JEKYLL_KEY),
        )
        default_state["updated_at"] = str(raw_state.get("updated_at", now))
        if default_state["active_character"] not in CHARACTER_CONFIGS:
            default_state["active_character"] = DEFAULT_ACTIVE_CHARACTER
        return default_state

    silicone_payload = {
        "character_key": SILICONE_KEY,
        "title": raw_state.get("title", CHARACTER_CONFIGS[SILICONE_KEY]["title"]),
        "conversation_id": raw_state.get("conversation_id", ""),
        "shift": raw_state.get("shift", CHARACTER_CONFIGS[SILICONE_KEY]["default_shift"]),
        "mood": raw_state.get("mood"),
        "active_memories": raw_state.get("active_memories", []),
        "messages": raw_state.get("messages", []),
        "created_at": raw_state.get("created_at", now),
        "updated_at": raw_state.get("updated_at", now),
    }
    default_state["characters"][SILICONE_KEY] = _normalise_character_state(
        SILICONE_KEY,
        silicone_payload,
    )
    default_state["updated_at"] = str(raw_state.get("updated_at", now))
    return default_state


def _state_for_client(state: dict[str, Any]) -> dict[str, Any]:
    return {
        "active_character": state["active_character"],
        "characters": {
            key: {
                **character_state,
                "label": CHARACTER_CONFIGS[key]["label"],
                "empty_message": CHARACTER_CONFIGS[key]["empty_message"],
                "subtitle": CHARACTER_CONFIGS[key]["subtitle"],
                "media": _discover_character_media(key),
            }
            for key, character_state in state["characters"].items()
        },
        "updated_at": state["updated_at"],
    }


def _load_state(store: ChatStateStore) -> dict[str, Any]:
    raw = store.load()
    state = _migrate_legacy_state(raw)
    if raw != state:
        store.save(state)
    return state


def _gather_pdf_paths(*folders: Path) -> list[Path]:
    paths: list[Path] = []
    for folder in folders:
        if not folder.exists():
            continue
        paths.extend(sorted(folder.rglob("*.pdf")))
    return paths


def _build_context_block(chunks: list[dict[str, object]]) -> str:
    if not chunks:
        return "No retrieved context was available for this turn."

    return "\n\n---\n\n".join(
        (
            f"[Chunk {index + 1} | persona {chunk['persona']} | score {chunk['score']:.3f}]\n"
            f"Source: {chunk['source_path']}\n"
            f"{chunk['text']}"
        )
        for index, chunk in enumerate(chunks)
    )


def _discover_character_media(character_key: str) -> dict[str, object]:
    folder = STATIC_MEDIA_ROOT / character_key
    images: list[dict[str, str]] = []
    videos: list[dict[str, str]] = []
    if folder.exists():
        for path in sorted(item for item in folder.iterdir() if item.is_file()):
            ext = path.suffix.lower()
            relative_path = path.relative_to(APP_ROOT / "static").as_posix()
            entry = {
                "name": path.stem.replace("_", " ").replace("-", " "),
                "url": f"/static/{quote(relative_path, safe='/')}",
            }
            if ext in IMAGE_EXTENSIONS:
                images.append(entry)
            elif ext in VIDEO_EXTENSIONS:
                videos.append(entry)

    return {
        "images": images,
        "videos": videos,
        "folder_hint": f"src/pico_chatgpt_bridge_web/static/character_media/{character_key}/",
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

    silicone_paths = _gather_pdf_paths(PDF_ROOT / "angela_carter", PDF_ROOT / "housewife")
    jekyll_paths = _gather_pdf_paths(PDF_ROOT / "jekyll_hyde")

    rag_registry: dict[str, RAGEngine | None] = {
        SILICONE_KEY: None,
        JEKYLL_KEY: None,
    }
    rag_errors: dict[str, str | None] = {
        SILICONE_KEY: None,
        JEKYLL_KEY: None,
    }

    for key, paths in ((SILICONE_KEY, silicone_paths), (JEKYLL_KEY, jekyll_paths)):
        try:
            rag = RAGEngine(client=client, chat_model=model, pdf_paths=paths)
            rag.ensure_ready()
            rag_registry[key] = rag
        except Exception as exc:
            rag_errors[key] = str(exc)

    @app.get("/")
    def index() -> str:
        return render_template("index.html", model=model, characters=CHARACTER_CONFIGS)

    @app.get("/api/health")
    def health() -> object:
        return jsonify(
            {
                "ok": True,
                "model": model,
                "characters": {
                    key: {
                        "label": config["label"],
                        "ready": rag_registry[key] is not None,
                        "error": rag_errors[key],
                        "sources": (
                            rag_registry[key].describe_sources()
                            if rag_registry[key] is not None
                            else "Unavailable"
                        ),
                    }
                    for key, config in CHARACTER_CONFIGS.items()
                },
            }
        )

    @app.get("/api/chat/state")
    def get_state() -> object:
        state = _load_state(store)
        return jsonify(_state_for_client(state))

    @app.patch("/api/chat/state")
    def update_state() -> object:
        state = _load_state(store)
        payload = request.get_json(silent=True) or {}

        active_character = str(payload.get("active_character", state["active_character"]))
        if active_character not in CHARACTER_CONFIGS:
            return jsonify({"error": "Unknown character key."}), 400

        state["active_character"] = active_character
        character_state = state["characters"][active_character]

        if "conversation_id" in payload:
            character_state["conversation_id"] = str(payload.get("conversation_id", "")).strip()

        if active_character == SILICONE_KEY and "shift" in payload:
            character_state["shift"] = clamp_shift(payload.get("shift"))

        if "mood" in payload:
            mood_value = payload.get("mood")
            character_state["mood"] = str(mood_value).strip() if mood_value else None

        if "active_memories" in payload and isinstance(payload["active_memories"], list):
            character_state["active_memories"] = [
                item for item in payload["active_memories"] if isinstance(item, str)
            ]

        character_state["updated_at"] = _utc_now()
        state["updated_at"] = character_state["updated_at"]
        store.save(state)
        return jsonify(_state_for_client(state))

    @app.post("/api/chat/state/reset")
    def reset_state() -> object:
        state = _load_state(store)
        payload = request.get_json(silent=True) or {}
        character_key = str(payload.get("character_key", state["active_character"]))
        if character_key not in CHARACTER_CONFIGS:
            return jsonify({"error": "Unknown character key."}), 400

        existing = state["characters"][character_key]
        preserved_conversation_id = str(existing.get("conversation_id", "") or "")
        state["characters"][character_key] = _new_character_state(
            character_key,
            conversation_id=preserved_conversation_id,
            shift=existing.get("shift"),
        )
        state["updated_at"] = state["characters"][character_key]["updated_at"]
        store.save(state)
        return jsonify(_state_for_client(state))

    @app.post("/api/chat/randomize-mood")
    def randomize_mood() -> object:
        state = _load_state(store)
        character_state = state["characters"][SILICONE_KEY]
        character_state["mood"] = choose_random_mood(rng)
        character_state["updated_at"] = _utc_now()
        state["updated_at"] = character_state["updated_at"]
        store.save(state)
        return jsonify(_state_for_client(state))

    @app.post("/api/chat/add-memory")
    def add_memory() -> object:
        state = _load_state(store)
        character_state = state["characters"][SILICONE_KEY]
        memory = choose_random_memory(character_state.get("active_memories", []), rng)
        if not memory:
            return jsonify({"error": "No memory files were found in data/."}), 400

        memories = list(character_state.get("active_memories", []))
        memories.append(memory)
        character_state["active_memories"] = memories[-5:]
        character_state["updated_at"] = _utc_now()
        state["updated_at"] = character_state["updated_at"]
        store.save(state)
        return jsonify({"memory": memory, "state": _state_for_client(state)})

    @app.post("/api/chat")
    def create_message() -> object:
        try:
            state = _load_state(store)
            payload = request.get_json(silent=True) or {}
            user_text = str(payload.get("message", "")).strip()
            if not user_text:
                return jsonify({"error": "Message is required."}), 400

            character_key = str(payload.get("character_key", state["active_character"]))
            if character_key not in CHARACTER_CONFIGS:
                return jsonify({"error": "Unknown character key."}), 400

            rag = rag_registry[character_key]
            rag_error = rag_errors[character_key]
            if rag is None:
                return jsonify({"error": rag_error or "RAG is not ready for this character."}), 400

            character_state = state["characters"][character_key]
            shift = clamp_shift(character_state.get("shift", 0.0))
            mood = character_state.get("mood")
            active_memories = list(character_state.get("active_memories", []))

            if character_key == SILICONE_KEY:
                chunks = rag.retrieve(user_text, shift=shift)
                instructions = build_shift_instruction(shift)
                live_instruction = build_live_instruction(mood, active_memories)
            else:
                chunks = rag.retrieve(user_text)
                instructions = JEKYLL_HYDE_INSTRUCTIONS
                live_instruction = ""

            retrieved_chunks = [_serialise_chunk(chunk) for chunk in chunks]
            context_block = _build_context_block(retrieved_chunks)
            input_parts = [f"Context:\n{context_block}", f"Question: {user_text}"]
            if live_instruction:
                input_parts.append(live_instruction)

            request_args: dict[str, object] = {
                "model": model,
                "instructions": instructions,
                "input": "\n\n".join(input_parts),
            }

            conversation_id = str(character_state.get("conversation_id", "") or "").strip()
            if conversation_id:
                request_args["conversation"] = conversation_id

            response = client.responses.create(**request_args)
            assistant_text = _extract_response_text(response) or "[No text output returned.]"
            response_conversation = getattr(response, "conversation", None)
            if response_conversation is not None and getattr(response_conversation, "id", ""):
                character_state["conversation_id"] = str(response_conversation.id)

            messages = list(character_state.get("messages", []))
            user_message = {
                "role": "user",
                "content": user_text,
                "created_at": _utc_now(),
            }
            assistant_message = {
                "role": "assistant",
                "content": assistant_text,
                "response_id": getattr(response, "id", ""),
                "retrieved_context": retrieved_chunks,
                "created_at": _utc_now(),
            }
            messages.extend([user_message, assistant_message])
            character_state["messages"] = messages
            character_state["updated_at"] = _utc_now()
            state["active_character"] = character_key
            state["updated_at"] = character_state["updated_at"]
            store.save(state)

            return jsonify(
                {
                    "state": _state_for_client(state),
                    "assistant": assistant_message,
                    "character_key": character_key,
                }
            )
        except Exception as exc:
            return jsonify({"error": str(exc)}), 400

    @app.get("/api/chat/export")
    def export_state() -> object:
        state = _load_state(store)
        return app.response_class(
            response=json.dumps(state, indent=2, ensure_ascii=True),
            mimetype="application/json",
        )

    return app


app = create_app()


if __name__ == "__main__":
    app.run(debug=True)
