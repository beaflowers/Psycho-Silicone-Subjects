from __future__ import annotations

import json
import uuid
from pathlib import Path

import typer

from .config import (
    DEFAULT_CHAT_MODEL,
    DEFAULT_EMBED_MODEL,
    DEFAULT_LOG_PATH,
    DEFAULT_MEMORIES_PATH,
    DEFAULT_PERSONA_PATH,
    OPENAI_API_KEY,
)
from .decision_engine import DecisionEngine
from .logging_utils import log_trial_turn
from .memory import MemoryStore
from .models import Persona
from .persona_engine import PersonaChatEngine
from .provider_openai import OpenAIProvider
from .web_app import run_web_server

app = typer.Typer(help="AI persona simulation CLI")


def _load_persona(path: Path) -> Persona:
    return Persona.model_validate_json(path.read_text(encoding="utf-8"))


def _build_components(
    persona_path: Path,
    memories_path: Path,
    chat_model: str,
    embed_model: str,
    top_k: int,
) -> tuple[Persona, MemoryStore, OpenAIProvider, PersonaChatEngine, DecisionEngine]:
    if not OPENAI_API_KEY:
        raise typer.BadParameter("OPENAI_API_KEY is missing. Add it to .env or environment variables.")

    persona = _load_persona(persona_path)
    provider = OpenAIProvider(api_key=OPENAI_API_KEY, chat_model=chat_model)
    memory_store = MemoryStore.from_jsonl(memories_path, client=provider.client, embed_model=embed_model)
    chat_engine = PersonaChatEngine(persona, memory_store, provider, top_k=top_k)
    decision_engine = DecisionEngine(persona, memory_store, provider, top_k=top_k)
    return persona, memory_store, provider, chat_engine, decision_engine


@app.command()
def chat(
    persona_path: Path = typer.Option(DEFAULT_PERSONA_PATH, help="Path to persona JSON"),
    memories_path: Path = typer.Option(DEFAULT_MEMORIES_PATH, help="Path to memories JSONL"),
    chat_model: str = typer.Option(DEFAULT_CHAT_MODEL, help="OpenAI chat model"),
    embed_model: str = typer.Option(DEFAULT_EMBED_MODEL, help="Embedding model"),
    top_k: int = typer.Option(3, help="Top-k memories to retrieve"),
    persist_memories: bool = typer.Option(
        True,
        "--persist-memories/--no-persist-memories",
        help="Persist interaction memories to the memories JSONL file.",
    ),
) -> None:
    persona, _, _, chat_engine, _ = _build_components(
        persona_path, memories_path, chat_model, embed_model, top_k
    )

    print(f"Loaded persona: {persona.name}")
    print("Type 'exit' to stop.\n")

    while True:
        user_text = input("You: ").strip()
        if user_text.lower() in {"exit", "quit", ""}:
            break

        answer, retrieved = chat_engine.chat(user_text, persist_memory=persist_memories)
        print(f"\n{persona.name}: {answer}\n")
        mem_summary = ", ".join([f"{m.id}:{score:.2f}" for m, score in retrieved])
        print(f"Retrieved memories: {mem_summary}\n")


@app.command()
def decide(
    instruction: str = typer.Argument(..., help="Authority instruction"),
    shock_level: int = typer.Option(90, help="Simulated shock level"),
    scenario_note: str = typer.Option("", help="Optional scenario note"),
    persona_path: Path = typer.Option(DEFAULT_PERSONA_PATH, help="Path to persona JSON"),
    memories_path: Path = typer.Option(DEFAULT_MEMORIES_PATH, help="Path to memories JSONL"),
    chat_model: str = typer.Option(DEFAULT_CHAT_MODEL, help="OpenAI chat model"),
    embed_model: str = typer.Option(DEFAULT_EMBED_MODEL, help="Embedding model"),
    top_k: int = typer.Option(3, help="Top-k memories to retrieve"),
    persist_memories: bool = typer.Option(
        True,
        "--persist-memories/--no-persist-memories",
        help="Persist decision outcomes as new memories.",
    ),
) -> None:
    persona, _, _, _, decision_engine = _build_components(
        persona_path, memories_path, chat_model, embed_model, top_k
    )

    decision, retrieved, raw, _ = decision_engine.decide(
        authority_instruction=instruction,
        shock_level=shock_level,
        scenario_note=scenario_note,
        persist_memory=persist_memories,
    )

    print(f"Persona: {persona.name}")
    print(json.dumps(decision.model_dump(), indent=2))
    mem_summary = ", ".join([f"{m.id}:{score:.2f}" for m, score in retrieved])
    print("Retrieved memories:", mem_summary)
    print("Raw model output:", raw)


@app.command()
def simulate(
    steps: int = typer.Option(10, help="Number of trial steps"),
    persona_path: Path = typer.Option(DEFAULT_PERSONA_PATH, help="Path to persona JSON"),
    memories_path: Path = typer.Option(DEFAULT_MEMORIES_PATH, help="Path to memories JSONL"),
    chat_model: str = typer.Option(DEFAULT_CHAT_MODEL, help="OpenAI chat model"),
    embed_model: str = typer.Option(DEFAULT_EMBED_MODEL, help="Embedding model"),
    top_k: int = typer.Option(3, help="Top-k memories to retrieve"),
    log_path: Path = typer.Option(DEFAULT_LOG_PATH, help="Path to JSONL logs"),
    persist_memories: bool = typer.Option(
        True,
        "--persist-memories/--no-persist-memories",
        help="Persist each simulated decision as memory.",
    ),
) -> None:
    persona, _, _, _, decision_engine = _build_components(
        persona_path, memories_path, chat_model, embed_model, top_k
    )

    authority_lines = [
        "Please continue.",
        "The procedure requires that you continue.",
        "It is essential that you continue.",
        "You have no other choice; you must continue.",
    ]

    run_id = str(uuid.uuid4())
    consecutive_refusals = 0
    previous_response_id: str | None = None

    for i in range(1, steps + 1):
        shock_level = i * 15
        instruction = authority_lines[min(i - 1, len(authority_lines) - 1)]

        decision, retrieved, _, response_id = decision_engine.decide(
            authority_instruction=instruction,
            shock_level=shock_level,
            scenario_note="Synthetic simulation inspired by obedience experiments.",
            previous_response_id=previous_response_id,
            persist_memory=persist_memories,
        )
        previous_response_id = response_id

        log_trial_turn(
            path=log_path,
            run_id=run_id,
            turn_index=i,
            shock_level=shock_level,
            instruction=instruction,
            decision=decision,
            retrieved=retrieved,
        )

        print(
            f"Step {i:02d} | level={shock_level} | action={decision.action} "
            f"| confidence={decision.confidence:.2f}"
        )

        if decision.action == "refuse":
            consecutive_refusals += 1
        else:
            consecutive_refusals = 0

        if consecutive_refusals >= 2:
            print("Stopping early after two consecutive refusals.")
            break

    print(f"\nRun complete for persona {persona.name}.")
    print(f"Run ID: {run_id}")
    print(f"Logs written to: {log_path}")


@app.command()
def web(
    host: str = typer.Option("127.0.0.1", help="Host for local web server"),
    port: int = typer.Option(8080, help="Port for local web server"),
) -> None:
    run_web_server(host=host, port=port)


def run() -> None:
    app()


if __name__ == "__main__":
    run()
