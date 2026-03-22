"""Prompt-building helpers."""

from __future__ import annotations


def build_prompt(button_value: str) -> str:
    """Build a simple prompt from the incoming button value."""
    return (
        "You are receiving input from an external button.\n"
        f"The button input is: {button_value}\n"
        "If the  button is 'blue', you respond with more meloncholy and introspective text.\n"
        "If the button is 'yellow', you respond with more cheerful and optimistic text.\n"
        "If the button is 'red', you respond with more angry and aggressive text.\n"
        "If the button is 'green', you respond with more calm and peaceful text.\n"
        "Write a short response (1-3 sentences) that reflects the tone of the button input.\n"
    )
