from __future__ import annotations

import importlib.util
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI


PERSONA_JEKYLL = "jekyllhyde"
PERSONA_FEMWIFE = "femwife"


@dataclass
class PersonaRuntimeState:
    previous_response_id: str | None = None
    shift: float = 0.5


class PersonaOrchestrator:
    """Adapter layer that proxies prompts to the two existing RAG implementations."""

    def __init__(self, workspace_root: Path) -> None:
        self.workspace_root = workspace_root
        self._load_env()
        api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is missing. Add it to a .env file.")

        self.client = OpenAI(api_key=api_key)
        self.chat_model = os.environ.get("OPENAI_CHAT_MODEL", "gpt-5-mini")

        self.jekyll_engine = self._build_jekyll_engine()
        self.femwife_engine = self._build_femwife_engine()

    def _load_env(self) -> None:
        env_candidates = [
            self.workspace_root / ".env",
            self.workspace_root / "jekyllandhyde" / ".env",
            self.workspace_root / "app" / "femandhousewife" / ".env",
            self.workspace_root / "milgram_experiment" / ".env",
        ]
        for env_path in env_candidates:
            if env_path.exists():
                load_dotenv(env_path, override=False)

    def _build_jekyll_engine(self) -> Any:
        module_path = self.workspace_root / "jekyllandhyde" / "app" / "rag_engine.py"
        if not module_path.exists():
            raise RuntimeError(f"Missing Jekyll RAG file: {module_path}")

        module_name = "jekyll_rag_module"
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            raise RuntimeError("Could not import Jekyll RAG engine.")

        module = importlib.util.module_from_spec(spec)
        # Register module before exec so decorators (like @dataclass) can
        # resolve cls.__module__ via sys.modules during import time.
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

        jekyll_pdf = self._find_jekyll_pdf()
        model = os.environ.get("MILGRAM_JEKYLL_MODEL", self.chat_model)
        return module.RAGEngine(client=self.client, pdf_paths=[jekyll_pdf], chat_model=model)

    def _build_femwife_engine(self) -> Any:
        legacy_src_root = self.workspace_root / "app" / "femandhousewife" / "src"
        src_root = self.workspace_root / "src"
        if not src_root.exists() and legacy_src_root.exists():
            src_root = legacy_src_root
        if not src_root.exists():
            raise RuntimeError(f"Missing Pico src folder: {src_root}")

        if str(src_root) not in sys.path:
            sys.path.insert(0, str(src_root))

        from pico_chatgpt_bridge.rag_engine import RAGEngine as PicoRAGEngine

        model = os.environ.get("MILGRAM_FEMWIFE_MODEL", self.chat_model)
        return PicoRAGEngine(
            client=self.client,
            chat_model=model,
            pdf_paths=self._find_femwife_pdfs(),
            top_k=int(os.environ.get("RAG_TOP_K", "5")),
        )

    def _find_jekyll_pdf(self) -> Path:
        candidate = (
            self.workspace_root
            / "jekyllandhyde"
            / "DrJekyllandMrHydePDF"
            / "The Strange Case of DrJekyll and Mr. Hyde.pdf"
        )
        if candidate.exists():
            return candidate

        fallback = sorted((self.workspace_root / "jekyllandhyde").rglob("*.pdf"))
        if not fallback:
            raise RuntimeError("No Jekyll/Hyde PDF found in jekyllandhyde folder.")
        return fallback[0]

    def _find_femwife_pdfs(self) -> list[Path]:
        legacy_folder = self.workspace_root / "app" / "femandhousewife" / "PDFs"
        repo_pdf_dirs = [
            self.workspace_root / "PDFs" / "angela_carter",
            self.workspace_root / "PDFs" / "housewife",
        ]

        if legacy_folder.exists():
            pdfs = sorted(
                path
                for path in legacy_folder.rglob("*.pdf")
                if any(token in str(path).lower() for token in ("angela", "housewife"))
            )
        else:
            pdfs = sorted(
                path
                for folder in repo_pdf_dirs
                if folder.exists()
                for path in folder.rglob("*.pdf")
            )

        if not pdfs:
            raise RuntimeError(
                "No femwife PDFs found in app/femandhousewife/PDFs or PDFs/{angela_carter,housewife}."
            )
        return pdfs

    def describe_sources(self) -> dict[str, str]:
        return {
            PERSONA_JEKYLL: self.jekyll_engine.describe_sources(),
            PERSONA_FEMWIFE: self.femwife_engine.describe_sources(),
        }

    def ask(
        self,
        persona_key: str,
        prompt: str,
        state: PersonaRuntimeState,
        *,
        top_k: int | None = None,
        forced_shift: float | None = None,
    ) -> dict[str, Any]:
        if persona_key == PERSONA_JEKYLL:
            answer, response_id, chunks = self.jekyll_engine.ask(
                prompt,
                previous_response_id=state.previous_response_id,
                k=top_k,
            )
            state.previous_response_id = response_id
            return {
                "answer": answer,
                "response_id": response_id,
                "metadata": {
                    "retrieval": [
                        {
                            "chunk_id": f"chunk_{index + 1}",
                            "path": chunk.source_path,
                            "page": chunk.page_number,
                            "score": round(float(chunk.score), 4),
                            "excerpt": self._short_excerpt(chunk.text),
                        }
                        for index, chunk in enumerate(chunks)
                    ]
                },
            }

        if persona_key == PERSONA_FEMWIFE:
            shift = forced_shift if forced_shift is not None else self._auto_shift(state.shift, prompt)
            tone_instruction = (
                "You are in a live experiment. Let your tone adapt naturally to pressure, "
                "authority, empathy, and tension in the conversation while staying grounded "
                "in retrieved context."
            )
            answer, response_id, chunks = self._ask_femwife_compatible(
                prompt,
                previous_response_id=state.previous_response_id,
                tone_instruction=tone_instruction,
                shift=shift,
            )
            state.previous_response_id = response_id
            state.shift = shift
            return {
                "answer": answer,
                "response_id": response_id,
                "metadata": {
                    "shift": round(shift, 3),
                    "retrieval": [
                        {
                            "chunk_id": f"chunk_{index + 1}",
                            "path": chunk.source_path,
                            "page": None,
                            "persona": chunk.persona,
                            "score": round(float(chunk.score), 4),
                            "excerpt": self._short_excerpt(chunk.text),
                        }
                        for index, chunk in enumerate(chunks)
                    ],
                },
            }

        raise ValueError(f"Unknown persona key: {persona_key}")

    def _ask_femwife_compatible(
        self,
        question: str,
        *,
        previous_response_id: str | None,
        tone_instruction: str,
        shift: float,
    ) -> tuple[str, str, list[Any]]:
        """
        Call Pico RAG with model-compatible request args.
        gpt-4.1 rejects reasoning/text controls used by some GPT-5 models.
        """
        from pico_chatgpt_bridge.prompting import (
            build_shift_instruction,
            clamp_shift,
            describe_shift,
        )
        from pico_chatgpt_bridge.rag_engine import _extract_response_text

        clamped_shift = clamp_shift(shift)
        chunks = self.femwife_engine.retrieve(question, shift=clamped_shift)
        context = "\n\n---\n\n".join(
            (
                f"[Chunk {index + 1} | persona {chunk.persona} | score {chunk.score:.3f}]\n"
                f"Source: {chunk.source_path}\n"
                f"{chunk.text.strip()}"
            )
            for index, chunk in enumerate(chunks)
        )

        input_parts = [
            f"Persona blend: {describe_shift(clamped_shift)}",
            f"Context:\n{context}",
            f"Question: {question}",
        ]
        if tone_instruction:
            input_parts.append(tone_instruction)

        model = getattr(self.femwife_engine, "_chat_model", self.chat_model)
        request_args: dict[str, Any] = {
            "model": model,
            "instructions": build_shift_instruction(clamped_shift),
            "input": "\n\n".join(input_parts),
            "max_output_tokens": 1200,
        }
        if previous_response_id:
            request_args["previous_response_id"] = previous_response_id

        # Only pass these controls for GPT-5 family.
        if str(model).startswith("gpt-5"):
            reasoning_effort = os.environ.get("OPENAI_REASONING_EFFORT", "minimal")
            text_verbosity = os.environ.get("OPENAI_TEXT_VERBOSITY", "low")
            request_args["reasoning"] = {"effort": reasoning_effort}
            request_args["text"] = {"verbosity": text_verbosity}

        response = self.client.responses.create(**request_args)
        response_text = _extract_response_text(response)
        if not response_text:
            raise RuntimeError("Femwife model returned no text output.")
        return response_text, response.id, chunks

    @staticmethod
    def _short_excerpt(text: str, limit: int = 180) -> str:
        compact = " ".join(str(text).split())
        if len(compact) <= limit:
            return compact
        return compact[: limit - 3].rstrip() + "..."

    def _auto_shift(self, previous: float, prompt: str) -> float:
        """Small heuristic to evolve Pico shift from conversation pressure."""
        lowered = prompt.lower()
        stress_terms = (
            "shock",
            "authority",
            "pain",
            "distress",
            "fear",
            "urgent",
            "refuse",
            "obey",
            "danger",
            "critical",
        )
        soothe_terms = (
            "calm",
            "safe",
            "gentle",
            "thank",
            "pause",
            "reflect",
            "comfort",
            "breathe",
        )

        stress_score = sum(1 for term in stress_terms if term in lowered)
        soothe_score = sum(1 for term in soothe_terms if term in lowered)

        delta = (stress_score * 0.06) - (soothe_score * 0.04)
        if "!" in prompt:
            delta += 0.02

        # Gentle drift back toward center so the shift is dynamic across time.
        drift = 0.01 if previous < 0.5 else -0.01
        updated = previous + delta + drift
        return max(0.0, min(1.0, round(updated, 3)))
