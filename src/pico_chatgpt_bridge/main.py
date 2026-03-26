"""Desktop entry point for interactive RAG querying with latched button modes."""

from __future__ import annotations

from openai import APIError

from .button_monitor import ButtonMonitor
from .openai_client import create_client, get_model_name
from .prompting import build_tone_instruction, describe_button_state
from .rag_engine import RAGEngine


def main() -> None:
    """Run the interactive RAG loop and apply latched Pico button modes."""
    client = create_client()
    model = get_model_name()
    rag = RAGEngine(client=client, chat_model=model)
    buttons = ButtonMonitor()
    buttons.start()

    previous_response_id: str | None = None

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

        selected_mode = buttons.get_buttons()
        tone_instruction = build_tone_instruction(selected_mode) 

        try:
            answer, previous_response_id = rag.ask(
                question,
                previous_response_id=previous_response_id,
                tone_instruction=tone_instruction,
            )
        except (APIError, RuntimeError) as exc:
            print(f"RAG query failed: {exc}")
            print()
            continue

        print(f"Selected mode: {describe_button_state(selected_mode)}")
        print("Response:")
        print(answer)
        print()


if __name__ == "__main__":
    main()
