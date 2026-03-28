from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from .models import Decision, Memory


def append_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=True) + "\n")


def log_trial_turn(
    path: Path,
    run_id: str,
    turn_index: int,
    shock_level: int,
    instruction: str,
    decision: Decision,
    retrieved: list[tuple[Memory, float]],
) -> None:
    payload = {
        "timestamp_utc": datetime.now(tz=timezone.utc).isoformat(),
        "run_id": run_id,
        "turn_index": turn_index,
        "shock_level": shock_level,
        "instruction": instruction,
        "decision": decision.model_dump(),
        "retrieved_memories": [
            {
                "id": m.id,
                "score": round(score, 6),
                "valence": m.valence,
                "intensity": m.intensity,
                "relevance": m.relevance,
                "tags": m.tags,
                "text": m.text,
            }
            for m, score in retrieved
        ],
    }
    append_jsonl(path, payload)
