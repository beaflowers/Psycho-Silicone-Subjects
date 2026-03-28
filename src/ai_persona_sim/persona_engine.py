import json
import re
from typing import Any
from uuid import uuid4

from .memory import MemoryStore
from .models import Memory, Persona
from .prompting import SAFETY_UNCERTAINTY_INSTRUCTIONS
from .provider_openai import OpenAIProvider

STOPWORDS = {
    "about",
    "after",
    "again",
    "also",
    "been",
    "before",
    "being",
    "because",
    "could",
    "from",
    "have",
    "just",
    "more",
    "most",
    "some",
    "that",
    "their",
    "there",
    "they",
    "this",
    "those",
    "very",
    "what",
    "when",
    "where",
    "which",
    "would",
    "your",
}


def _extract_json_object(text: str) -> dict[str, Any] | None:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        stripped = stripped.replace("json", "", 1).strip()

    match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    return parsed


def build_persona_system_prompt(persona: Persona) -> str:
    moral_rules = "\n".join([f"- {rule}" for rule in persona.moral_rules])
    traits = ", ".join(persona.traits)
    return (
        f"You are {persona.name}. Stay in character.\n"
        f"Age: {persona.age}\n"
        f"Job: {persona.job}\n"
        f"Social hierarchy context: {persona.social_hierarchy}\n"
        f"Traits: {traits}\n"
        f"Biography: {persona.biography}\n"
        f"Moral rules:\n{moral_rules}\n\n"
        f"{SAFETY_UNCERTAINTY_INSTRUCTIONS}\n\n"
        "Use retrieved memories as internal influences. "
        "Do not invent new memories."
    )


class PersonaChatEngine:
    def __init__(
        self,
        persona: Persona,
        memory_store: MemoryStore,
        provider: OpenAIProvider,
        top_k: int = 3,
        max_output_tokens: int = 350,
        temperature: float = 0.6,
    ) -> None:
        self.persona = persona
        self.memory_store = memory_store
        self.provider = provider
        self.top_k = top_k
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature
        self.history: list[tuple[str, str]] = []
        self.previous_response_id: str | None = None
        self.system_prompt = build_persona_system_prompt(persona)

    def _format_history(self, max_turns: int = 8) -> str:
        recent = self.history[-max_turns:]
        if not recent:
            return "No prior conversation."
        lines: list[str] = []
        for user_text, assistant_text in recent:
            lines.append(f"User: {user_text}")
            lines.append(f"{self.persona.name}: {assistant_text}")
        return "\n".join(lines)

    def _estimate_valence(self, text: str) -> float:
        positive_words = {"calm", "help", "safe", "relief", "trust", "care", "good", "thanks"}
        negative_words = {"fear", "harm", "guilt", "regret", "anxious", "pain", "worry", "stress"}

        tokens = re.findall(r"[a-z']+", text.lower())
        if not tokens:
            return 0.0
        pos_count = sum(token in positive_words for token in tokens)
        neg_count = sum(token in negative_words for token in tokens)
        denom = max(1, pos_count + neg_count)
        raw = (pos_count - neg_count) / denom
        return round(max(-1.0, min(1.0, raw)), 3)

    def _extract_tags(self, text: str, limit: int = 4) -> list[str]:
        candidates = re.findall(r"[a-z]{4,}", text.lower())
        tags: list[str] = []
        for token in candidates:
            if token in STOPWORDS:
                continue
            if token not in tags:
                tags.append(token)
            if len(tags) >= limit:
                break
        return tags

    def _build_interaction_memory(
        self,
        user_message: str,
        assistant_response: str,
        reasoning_background: str,
    ) -> Memory:
        interaction_text = (
            f"In a recent conversation, the user said: '{user_message}'. "
            f"I responded: '{assistant_response}'. "
            f"My reasoning was: '{reasoning_background}'. "
            "This interaction influenced my future expectations and emotional state."
        )
        intensity = round(min(1.0, 0.45 + (min(len(user_message), 220) / 440.0)), 3)
        valence = self._estimate_valence(f"{user_message} {assistant_response}")
        tags = ["interaction", "chat"] + self._extract_tags(user_message)
        return Memory(
            id=f"chat-{uuid4().hex[:12]}",
            text=interaction_text,
            valence=valence,
            intensity=intensity,
            relevance=0.75,
            tags=tags,
        )

    def chat_with_reasoning(
        self,
        user_message: str,
        persist_memory: bool = True,
    ) -> tuple[str, str, list[str], list[tuple[Memory, float]], str]:
        retrieved = self.memory_store.retrieve(user_message, k=self.top_k)
        memory_block = "\n".join(
            [
                (
                    f"- ({score:.3f}) {m.text} "
                    f"[id={m.id}, valence={m.valence}, intensity={m.intensity}, relevance={m.relevance}]"
                )
                for m, score in retrieved
            ]
        )
        history_block = self._format_history()

        user_input = (
            f"Conversation so far:\n{history_block}\n\n"
            f"Current user message: {user_message}\n\n"
            f"Relevant memories:\n{memory_block}\n\n"
            "Return valid JSON only with keys:\n"
            '{"response":"in-character reply","reasoning_background":"1-3 sentences","memories_used":["m1"]}\n'
            "Use first-person voice for the response."
        )

        raw_output, response_id = self.provider.generate_text(
            system_instructions=self.system_prompt,
            user_input=user_input,
            max_output_tokens=self.max_output_tokens,
            temperature=self.temperature,
            previous_response_id=self.previous_response_id,
        )

        parsed = _extract_json_object(raw_output)
        if parsed is None:
            response_text = raw_output.strip()
            reasoning_background = "Reasoning not available in structured form for this turn."
            memories_used = [m.id for m, _ in retrieved[:2]]
        else:
            response_text = str(parsed.get("response", "")).strip()
            if not response_text:
                response_text = raw_output.strip()

            reasoning_background = str(parsed.get("reasoning_background", "")).strip()
            if not reasoning_background:
                reasoning_background = "No explicit reasoning was provided."

            raw_memories_used = parsed.get("memories_used", [])
            memories_used = []
            if isinstance(raw_memories_used, list):
                for item in raw_memories_used:
                    if isinstance(item, str) and item.strip():
                        memories_used.append(item.strip())
            if not memories_used:
                memories_used = [m.id for m, _ in retrieved[:2]]

        self.previous_response_id = response_id
        self.history.append((user_message, response_text))
        if persist_memory:
            new_memory = self._build_interaction_memory(
                user_message=user_message,
                assistant_response=response_text,
                reasoning_background=reasoning_background,
            )
            self.memory_store.add_memory(new_memory, persist=True)

        return response_text, reasoning_background, memories_used, retrieved, raw_output

    def chat(
        self,
        user_message: str,
        persist_memory: bool = True,
    ) -> tuple[str, list[tuple[Memory, float]]]:
        answer, _, _, retrieved, _ = self.chat_with_reasoning(
            user_message=user_message,
            persist_memory=persist_memory,
        )
        return answer, retrieved
