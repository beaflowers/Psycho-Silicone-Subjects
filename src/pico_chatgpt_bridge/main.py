"""Desktop entry point for interactive RAG querying with latched button modes."""

from __future__ import annotations

from openai import APIError

from .button_monitor import ButtonMonitor
from .openai_client import create_client, get_model_name
from .prompting import (
    build_tone_instruction,
    clamp_shift,
    describe_shift,
    describe_button_state,
)
from .rag_engine import RAGEngine


def main() -> None:
    """Run the interactive RAG loop and apply latched Pico button modes."""
    client = create_client()
    model = get_model_name()
    rag = RAGEngine(client=client, chat_model=model)
    buttons = ButtonMonitor()
    buttons.start()

    previous_response_id: str | None = None
    shift = 0.0

    print("Interactive RAG mode is ready.")
    print(f"RAG sources: {rag.describe_sources()}")
    print("Preparing the RAG index before interactive mode starts...")
    try:
        rag.ensure_ready()
    except (APIError, RuntimeError) as exc:
        print(f"RAG setup failed: {exc}")
        return
    print("Press a Pico button once to select a persistent tone mode.")
    print("Type a question and press Enter. Type /reset to clear conversation memory.")
    print("Use /shift <0..1> to gradually move from Angela Carter to housewife voice.")
    print("Press Enter on an empty line, or type quit/exit, to stop.")

    while True:
        try:
            question = input("Question (or 'quit'): ").strip()
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
                shift = 0.0
                print(f"Shift updated. {describe_shift(shift)}")
                print()
                continue

            if requested_shift == "housewife":
                shift = 1.0
                print(f"Shift updated. {describe_shift(shift)}")
                print()
                continue
            try:
                shift = clamp_shift(float(requested_shift))
            except ValueError:
                print("Invalid shift value. Use a number from 0 to 1, or /shift angela, or /shift housewife.")
                print()
                continue
            print(f"Shift updated. {describe_shift(shift)}")
            print()
            continue

        selected_mode = buttons.get_buttons()
        tone_instruction = build_tone_instruction(selected_mode)

        try:
            answer, previous_response_id = rag.ask(
                question,
                previous_response_id=previous_response_id,
                tone_instruction=tone_instruction,
                shift=shift,
            )
        except (APIError, RuntimeError) as exc:
            print(f"RAG query failed: {exc}")
            print()
            continue

        print(f"Persona blend: {describe_shift(shift)}")
        print(f"Selected mode: {describe_button_state(selected_mode)}")
        print("Response:")
        print(answer)
        print()


if __name__ == "__main__":
    main()
