"""Desktop entry point for interactive RAG querying with latched button modes."""

from __future__ import annotations

import random
import threading

from openai import APIError

from .button_monitor import ButtonMonitor
from .openai_client import create_client, get_model_name
from .prompting import (
    SHIFT_STEP,
    build_live_instruction,
    clamp_shift,
    choose_random_memory,
    choose_random_mood,
    describe_shift,
    step_shift,
)
from .rag_engine import RAGEngine

QUESTION_PROMPT = "Question (or 'quit'): "


def main() -> None:
    """Run the interactive RAG loop and apply live Pico button actions."""
    client = create_client()
    model = get_model_name()
    rag = RAGEngine(client=client, chat_model=model)
    rng = random.Random()
    state_lock = threading.Lock()

    previous_response_id: str | None = None
    shift = 0.0
    current_mood: str | None = None
    active_memories: list[str] = []

    def refresh_prompt() -> None:
        print(QUESTION_PROMPT, end="", flush=True)

    def handle_button_event(event: tuple[str, ...]) -> None:
        nonlocal shift, current_mood, active_memories

        with state_lock:
            for button in event:
                if button == "blue":
                    current_mood = choose_random_mood(rng)
                    print(f"Blue button: selected mood '{current_mood}'", flush=True)
                    refresh_prompt()
                    continue

                if button == "yellow":
                    shift = step_shift(shift, SHIFT_STEP)
                    print(
                        f"Yellow button: increased shift by {SHIFT_STEP:.1f}. {describe_shift(shift)}",
                        flush=True,
                    )
                    refresh_prompt()
                    continue

                if button == "red":
                    shift = step_shift(shift, -SHIFT_STEP)
                    print(
                        f"Red button: decreased shift by {SHIFT_STEP:.1f}. {describe_shift(shift)}",
                        flush=True,
                    )
                    refresh_prompt()
                    continue

                if button == "green":
                    memory = choose_random_memory(active_memories, rng)
                    if not memory:
                        print("Green button: no memory lines were found in data/.", flush=True)
                        refresh_prompt()
                        continue
                    active_memories.append(memory)
                    print(f"Green button: added memory '{memory}'", flush=True)
                    refresh_prompt()

    buttons = ButtonMonitor(on_event=handle_button_event)
    buttons.start()

    print("Interactive RAG mode is ready.")
    print(f"RAG sources: {rag.describe_sources()}")
    print("Preparing the RAG index before interactive mode starts...")
    try:
        rag.ensure_ready()
    except (APIError, RuntimeError) as exc:
        print(f"RAG setup failed: {exc}")
        return
    print("Blue randomizes the mood. Yellow raises shift by 0.1. Red lowers shift by 0.1. Green adds a memory.")
    print("Type a question and press Enter. Type /reset to clear conversation memory.")
    print("Use /shift <0..1> to gradually move from Angela Carter to housewife voice.")
    print("Press Enter on an empty line, or type quit/exit, to stop.")

    while True:
        try:
            question = input(QUESTION_PROMPT).strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting interactive mode.")
            break

        if question.lower() in {"quit", "exit", ""}:
            break

        if question == "/reset":
            previous_response_id = None
            print("Conversation memory cleared.")
            print()
            continue

        if question in {"/shift", "/shift current"}:
            print(f"Current {describe_shift(shift)}")
            print()
            continue

        if question.startswith("/shift"):
            _, _, requested_shift = question.partition(" ")
            requested_shift = requested_shift.strip().lower()
            if requested_shift in {"", "current"}:
                print(f"Current {describe_shift(shift)}")
                print()
                continue

            if requested_shift == "angela":
                with state_lock:
                    shift = 0.0
                print(f"Shift updated. {describe_shift(shift)}")
                print()
                continue

            if requested_shift == "housewife":
                with state_lock:
                    shift = 1.0
                print(f"Shift updated. {describe_shift(shift)}")
                print()
                continue
            try:
                parsed_shift = clamp_shift(float(requested_shift))
            except ValueError:
                print("Invalid shift value. Use a number from 0 to 1, or /shift angela, or /shift housewife.")
                print()
                continue
            with state_lock:
                shift = parsed_shift
            print(f"Shift updated. {describe_shift(shift)}")
            print()
            continue

        with state_lock:
            current_shift = shift
            mood_for_prompt = current_mood
            memories_for_prompt = list(active_memories)

        live_instruction = build_live_instruction(mood_for_prompt, memories_for_prompt)

        try:
            answer, previous_response_id = rag.ask(
                question,
                previous_response_id=previous_response_id,
                tone_instruction=live_instruction,
                shift=current_shift,
            )
        except (APIError, RuntimeError) as exc:
            print(f"RAG query failed: {exc}")
            print()
            continue

        print(f"Persona blend: {describe_shift(current_shift)}")
        print(f"Current mood: {mood_for_prompt or 'none'}")
        print(f"Active memory lines: {len(memories_for_prompt)}")
        print("Response:")
        print(answer)
        print()


if __name__ == "__main__":
    main()
