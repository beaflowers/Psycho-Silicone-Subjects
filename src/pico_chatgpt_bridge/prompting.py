"""Helpers for mapping Pico button input to live interaction controls."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
import json
from pathlib import Path
import random


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

BUTTON_ORDER = ("blue", "yellow", "red", "green")
SHIFT_MIN = 0.0
SHIFT_MAX = 1.0
SHIFT_STEP = 0.1
ANGELA_KEY = "angela_carter"
HOUSEWIFE_KEY = "housewife"
MOOD_OPTIONS = (
    "melancholy and introspective",
    "cheerful and optimistic",
    "angry and aggressive",
    "calm and peaceful",
)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MEMORY_DATA_DIR = PROJECT_ROOT / "data"
MEMORY_FILE_PATTERNS = (
    "*memories*.jsonl",
    "*memories*.json",
    "*memory*.jsonl",
    "*memory*.json",
)


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


def step_shift(shift: float | None, delta: float) -> float:
    """Apply a delta to the current shift and clamp the result."""
    baseline = SHIFT_MIN if shift is None else float(shift)
    return clamp_shift(baseline + delta)


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


def choose_random_mood(rng: random.Random | None = None) -> str:
    """Pick a random mood from the available live mood palette."""
    generator = rng or random
    return generator.choice(MOOD_OPTIONS)


def _iter_memory_files() -> tuple[Path, ...]:
    """Return matching memory data files from the local data directory."""
    if not MEMORY_DATA_DIR.exists():
        return ()

    matches: list[Path] = []
    seen: set[Path] = set()
    for pattern in MEMORY_FILE_PATTERNS:
        for path in sorted(MEMORY_DATA_DIR.glob(pattern)):
            resolved = path.resolve()
            if resolved in seen or not path.is_file():
                continue
            matches.append(path)
            seen.add(resolved)
    return tuple(matches)


def _extract_memory_texts(path: Path) -> list[str]:
    """Read one memory file and return the available memory text lines."""
    suffix = path.suffix.lower()
    raw_text = path.read_text(encoding="utf-8").strip()
    if not raw_text:
        return []

    def coerce_text(item: object) -> str | None:
        if isinstance(item, str):
            text = item.strip()
            return text or None
        if isinstance(item, dict):
            value = item.get("text")
            if isinstance(value, str):
                text = value.strip()
                return text or None
        return None

    memories: list[str] = []
    if suffix == ".jsonl":
        for line in raw_text.splitlines():
            if not line.strip():
                continue
            parsed = json.loads(line)
            text = coerce_text(parsed)
            if text:
                memories.append(text)
        return memories

    parsed = json.loads(raw_text)
    if isinstance(parsed, list):
        for item in parsed:
            text = coerce_text(item)
            if text:
                memories.append(text)
        return memories

    text = coerce_text(parsed)
    return [text] if text else []


def choose_random_memory(
    existing_memories: Iterable[str] = (),
    rng: random.Random | None = None,
) -> str:
    """Pick a random memory line, preferring ones not already active."""
    generator = rng or random
    existing = set(existing_memories)

    memory_options: list[str] = []
    for path in _iter_memory_files():
        memory_options.extend(_extract_memory_texts(path))

    if not memory_options:
        return ""

    available = [item for item in memory_options if item not in existing]
    if not available:
        available = list(memory_options)
    return generator.choice(available)


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
        "1.0 is fully a good-housewife instructional voice. "
        "Keep continuity with the ongoing conversation and let traits from earlier "
        "turns linger rather than resetting abruptly.\n\n"
        "For values under 0.5, author Angela Carter's voice is dominant and humerous. "
        "Above 0.5, she fights for control, interjecting with abjection and disapproval. The closer the shift gets to 1, the less she interjects."
        "Answer with wit and literary sharpness, referencing works and cultural "
        "touchstones she is familiar with. "
        "Angela has a strong love for the world and its people, and a critical eye "
        "for social dynamics and power structures, which can make her bitter. "
        "She is more bitter the higher the shift goes. "
        "She has a particualry fascination for fairy tales and cigarettes. "
        "Use the provided context as your primary grounding. "
        "If the context is incomplete, say so briefly instead of inventing details."
        "At values above 0.5, become more domestic, performing the role of an idealized "
        "mid-century good housewife. " 
        "She is practical, orderly, and demure, with a focus on domestic life and "
        "homemaking. "
        "She is proud of what she does, but her laughter comes from a place of fear, "
        "and her eyes are glassy with boredom. "
        "In the 0.4-0.6 range, the housewife is panicky, but still trying to maintain "
        "a facade of perfection, also more unhinged and desperate to maintain control "
        "over her domestic domain. "
        "Answer in 2-3 short sentences with crisp domestic confidence and practical "
        "instruction. "
        "As the shift approaches 1.0, the housewife becomes more confident and "
        "assertive, calming into her placid role. "
        "Use the provided housewife context as the primary grounding."
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


def build_live_instruction(
    mood: str | None,
    memory_items: Iterable[str],
) -> str:
    """Build a prompt modifier from the current mood and active memory lines."""
    parts: list[str] = []

    if mood:
        parts.append(
            "Current mood from the live controls: "
            f"{mood}. Let that emotional register color the reply while "
            "staying grounded in the retrieved material."
        )

    recent_memories = list(memory_items)[-3:]
    if recent_memories:
        memory_text = " ".join(f"- {item}" for item in recent_memories)
        parts.append(
            "Current memory fragments surfacing around the conversation. Treat them "
            f"as subjective traces that can influence the response when relevant: {memory_text}"
        )

    return "\n".join(parts)
