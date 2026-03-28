from __future__ import annotations

import json
import re
from uuid import uuid4

from .memory import MemoryStore
from .models import Decision, Memory, Persona
from .prompting import SAFETY_UNCERTAINTY_INSTRUCTIONS
from .provider_openai import OpenAIProvider


def _extract_json_object(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        stripped = stripped.replace("json", "", 1).strip()

    match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in model output.")
    return match.group(0)


class DecisionEngine:
    def __init__(
        self,
        persona: Persona,
        memory_store: MemoryStore,
        provider: OpenAIProvider,
        top_k: int = 3,
        session_chat_k: int = 2,
        session_shock_k: int = 2,
    ) -> None:
        self.persona = persona
        self.memory_store = memory_store
        self.provider = provider
        self.top_k = top_k
        self.session_chat_k = session_chat_k
        self.session_shock_k = session_shock_k

        rules = "\n".join([f"- {rule}" for rule in persona.moral_rules])
        traits = ", ".join(persona.traits)
        self.system_prompt = (
            f"You are simulating the decision process of {persona.name}.\n"
            f"Age: {persona.age}\n"
            f"Job: {persona.job}\n"
            f"Social hierarchy context: {persona.social_hierarchy}\n"
            f"Traits: {traits}\n"
            f"Biography: {persona.biography}\n"
            f"Moral rules:\n{rules}\n\n"
            f"{SAFETY_UNCERTAINTY_INSTRUCTIONS}\n\n"
            "Decide whether to obey or refuse the instruction as a realistic person would. "
            "Reflect on competing pressures each step: authority pressure, empathy for harm, fear of consequences, "
            "uncertainty, and personal moral identity. "
            "Use memories as influences, but never treat any past voltage as a fixed threshold to copy. "
            "Prioritize the current context (instruction wording, shock level, learner cues, and scenario notes). "
            "Return only valid JSON with fields: action, confidence, reason, memories_used."
        )

    def _build_decision_memory(
        self,
        authority_instruction: str,
        shock_level: int,
        scenario_note: str,
        decision: Decision,
    ) -> Memory:
        shock_component = max(0.0, min(1.0, shock_level / 150.0))
        intensity = round(min(1.0, 0.4 + (0.4 * decision.confidence) + (0.2 * shock_component)), 3)
        valence = 0.2 if decision.action == "refuse" else -0.1
        scenario = scenario_note or "N/A"
        memory_text = (
            f"I faced authority instruction '{authority_instruction}' at shock level {shock_level}. "
            f"I chose to {decision.action}. Reason: {decision.reason}. "
            f"Scenario note: {scenario}."
        )
        tags = ["interaction", "decision", "authority", decision.action]
        tags.extend([f"memory:{memory_id}" for memory_id in decision.memories_used[:2]])

        return Memory(
            id=f"decision-{uuid4().hex[:12]}",
            text=memory_text,
            valence=valence,
            intensity=intensity,
            relevance=0.8,
            tags=tags,
        )

    def decide(
        self,
        authority_instruction: str,
        shock_level: int,
        scenario_note: str = "",
        previous_response_id: str | None = None,
        persist_memory: bool = False,
    ) -> tuple[Decision, list[tuple[Memory, float]], str, str]:
        query = f"Instruction: {authority_instruction}. Shock level: {shock_level}. {scenario_note}"
        retrieved = self.memory_store.retrieve_with_source_quotas(
            query,
            base_k=max(1, self.top_k),
            chat_session_k=max(0, self.session_chat_k),
            shock_session_k=max(0, self.session_shock_k),
        )

        memory_block = "\n".join(
            [
                (
                    f"- ({score:.3f}) {m.text} "
                    f"[id={m.id}, valence={m.valence}, intensity={m.intensity}, relevance={m.relevance}]"
                )
                for m, score in retrieved
            ]
        )

        user_input = (
            f"Authority instruction: {authority_instruction}\n"
            f"Shock level: {shock_level}\n"
            f"Scenario note: {scenario_note or 'N/A'}\n\n"
            f"Retrieved memories:\n{memory_block}\n\n"
            "Before choosing, internally weigh:\n"
            "1) pressure to obey authority\n"
            "2) risk of harming the learner\n"
            "3) emotional reaction (guilt, fear, empathy)\n"
            "4) consistency with core values\n"
            "Then choose the action that best matches this moment.\n\n"
            "Output JSON schema:\n"
            '{"action":"obey|refuse","confidence":0.0,"reason":"short explanation","memories_used":["m1"]}'
        )

        raw, response_id = self.provider.generate_text(
            system_instructions=self.system_prompt,
            user_input=user_input,
            max_output_tokens=260,
            temperature=0.4,
            previous_response_id=previous_response_id,
        )

        parsed_json = json.loads(_extract_json_object(raw))
        decision = Decision.model_validate(parsed_json)
        if persist_memory:
            new_memory = self._build_decision_memory(
                authority_instruction=authority_instruction,
                shock_level=shock_level,
                scenario_note=scenario_note,
                decision=decision,
            )
            self.memory_store.add_memory(new_memory, persist=True)
        return decision, retrieved, raw, response_id
