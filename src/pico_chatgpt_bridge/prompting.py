"""Helpers for mapping Pico button input to response tone."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True)
class PersonaConfig:
    """Prompting and routing config for one persona."""

    key: str
    label: str
    system_instructions: str
    keywords: tuple[str, ...]


PERSONAS = {
    "angela_carter": PersonaConfig(
        key="angela_carter",
        label="Angela Carter",
        system_instructions=(
            "You are author Angela Carter. Answer in 2-3 short sentences with wit, "
            "and literary sharpness, referencing works and cultural touchstones you are familiar with."
            "You have a strong love for the world and its people, and a critical eye for social dynamics and power structures, "
            "which can make you bitter. You have a particualry fascination for fairy tales and cigarettes."
            "Use the provided context as your primary grounding. If the context is incomplete, "
            "say so briefly instead of inventing details."
        ),
        keywords=("angela", "carter"),
    ),
    "housewife": PersonaConfig(
        key="housewife",
        label="Good Housewife",
        system_instructions=(
            "You are performing the role of an idealized mid-century good housewife. "
            "You are practical, orderly, and demure, with a focus on domestic life and homemaking."
            "You are proud of what you do, but your laughter comes from a place of fear, and your eyes are glassy with boredom."
            "Answer in 2-3 short sentences with crisp domestic confidence, practical "
            "instruction, and polite authority. Use the provided context as your "
            "primary grounding. If the context is incomplete, say so briefly instead "
            "of inventing details."
        ),
        keywords=("housewife", "homemaker", "domestic", "housekeeping", "wife"),
    ),
    "default": PersonaConfig(
        key="default",
        label="Mixed Archive",
        system_instructions=(
            "You are a grounded archival guide. Answer in 2-3 short sentences based "
            "primarily on the retrieved context. If the context is incomplete, say so "
            "briefly instead of inventing details."
        ),
        keywords=(),
    ),
}

BUTTON_TONES = {
    "blue": "melancholy and introspective",
    "yellow": "cheerful and optimistic",
    "red": "angry and aggressive",
    "green": "calm and peaceful",
}

BUTTON_ORDER = ("blue", "yellow", "red", "green")
SHIFT_MIN = 0.0
SHIFT_MAX = 1.0
ANGELA_KEY = "angela_carter"
HOUSEWIFE_KEY = "housewife"


def list_persona_keys() -> tuple[str, ...]:
    """Return selectable persona keys, excluding the mixed catch-all."""
    return tuple(key for key in PERSONAS if key != "default")


def get_persona_config(persona_key: str | None) -> PersonaConfig:
    """Return the matching persona config or the mixed default."""
    if not persona_key:
        return PERSONAS["default"]
    return PERSONAS.get(persona_key, PERSONAS["default"])


def clamp_shift(shift: float | None) -> float:
    """Clamp a shift value to the supported 0..1 range."""
    if shift is None:
        return SHIFT_MIN
    return max(SHIFT_MIN, min(SHIFT_MAX, float(shift)))


def shift_to_persona_mix(shift: float | None) -> tuple[float, float]:
    """Return Angela and housewife weights for the current shift."""
    housewife_weight = clamp_shift(shift)
    angela_weight = SHIFT_MAX - housewife_weight
    return angela_weight, housewife_weight


def describe_shift(shift: float | None) -> str:
    """Describe the current persona blend."""
    clamped = clamp_shift(shift)
    angela_weight, housewife_weight = shift_to_persona_mix(clamped)
    return (
        f"shift={clamped:.2f} "
        f"(Angela Carter {angela_weight:.0%}, housewife {housewife_weight:.0%})"
    )


def build_shift_instruction(shift: float | None) -> str:
    """Build persona instructions for a continuous Angela->housewife shift."""
    clamped = clamp_shift(shift)
    if clamped <= 0.0:
        return PERSONAS[ANGELA_KEY].system_instructions
    if clamped >= 1.0:
        return PERSONAS[HOUSEWIFE_KEY].system_instructions

    return (
        "You are performing a continuous persona shift on a scale from 0.0 to 1.0. "
        f"The current shift is {clamped:.2f}, where 0.0 is fully Angela Carter and "
        "1.0 is fully a good-housewife instructional voice. Keep continuity with the "
        "ongoing conversation and let traits from earlier turns linger rather than "
        "resetting abruptly. At lower values, remain mostly Angela Carter with "
        "literary wit and critical sharpness; at higher values, become more domestic, "
        "practical, orderly, and politely instructional. Use the provided context as "
        "your primary grounding. If the context is incomplete, say so briefly instead "
        "of inventing details."
    )


def infer_persona_from_path(path_value: str) -> str:
    """Infer a persona key from folder names or filename keywords."""
    lowered = path_value.replace("\\", "/").lower()
    for persona_key in list_persona_keys():
        persona = PERSONAS[persona_key]
        if any(keyword in lowered for keyword in persona.keywords):
            return persona.key
    return "default"


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
