"""Helpers for mapping Pico button input to response tone."""

from __future__ import annotations

from collections.abc import Iterable

BUTTON_TONES = {
    "blue": "melancholy and introspective",
    "yellow": "cheerful and optimistic",
    "red": "angry and aggressive",
    "green": "calm and peaceful",
}

BUTTON_ORDER = ("blue", "yellow", "red", "green")


def parse_button_line(button_value: str) -> tuple[str, ...]:
    """Parse a serial button line into a stable, ordered tuple."""
    raw_buttons = {
        part.strip().lower()
        for part in button_value.split(",")
        if part.strip() and part.strip().lower() != "none"
    }
    return tuple(button for button in BUTTON_ORDER if button in raw_buttons)


def describe_button_state(buttons: Iterable[str]) -> str:
    """Return a compact human-readable button description."""
    button_list = list(buttons)
    if not button_list:
        return "none"
    return ", ".join(button_list)


def build_tone_instruction(buttons: Iterable[str]) -> str:
    """Convert active buttons into a tone instruction for the model."""
    tones = [BUTTON_TONES[button] for button in buttons if button in BUTTON_TONES]
    if not tones:
        return ""

    if len(tones) == 1:
        combined = tones[0]
    else:
        combined = ", ".join(tones[:-1]) + f", and {tones[-1]}"

    return (
        "Tone modifier from the live button state: shape the answer with a "
        f"{combined} voice while staying grounded in the retrieved material."
    )
