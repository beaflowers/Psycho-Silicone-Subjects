from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import faiss
import numpy as np

from .models import Memory


class MemoryStore:
    SEMANTIC_WEIGHT = 0.55
    RECENCY_WEIGHT = 0.25
    IMPORTANCE_WEIGHT = 0.20
    RECENCY_HALF_LIFE_DAYS = 30.0
    BASE_RECENCY_FALLBACK = 0.30
    SESSION_RECENCY_FALLBACK = 0.65

    def __init__(
        self,
        client,
        memories: list[Memory],
        embed_model: str,
        source_path: Path | None = None,
    ) -> None:
        self.client = client
        self.memories = memories
        self.embed_model = embed_model
        self.source_path = source_path
        self._index: faiss.IndexFlatIP | None = None
        self._build_index()

    @classmethod
    def from_jsonl(
        cls,
        path: Path,
        client,
        embed_model: str,
        *,
        filter_fn: Callable[[dict[str, Any]], bool] | None = None,
        default_source_type: str | None = None,
    ) -> "MemoryStore":
        memories: list[Memory] = []
        if not path.exists():
            raise FileNotFoundError(f"Memories file not found: {path}")
        for raw in path.read_text(encoding="utf-8").splitlines():
            if not raw.strip():
                continue
            parsed = json.loads(raw)
            if not isinstance(parsed, dict):
                continue
            if filter_fn is not None and not filter_fn(parsed):
                continue
            if default_source_type and "source_type" not in parsed:
                parsed["source_type"] = default_source_type
            memories.append(Memory.model_validate(parsed))
        return cls(client=client, memories=memories, embed_model=embed_model, source_path=path)

    def _memory_embedding_text(self, m: Memory) -> str:
        tags = ", ".join(m.tags)
        source_type = m.source_type or self._memory_source_type(m)
        importance = m.importance if m.importance is not None else self._importance_score(m)
        return (
            f"Memory: {m.text}\n"
            f"Source: {source_type}\n"
            f"Tags: {tags}\n"
            f"Valence: {m.valence}\n"
            f"Intensity: {m.intensity}\n"
            f"Relevance: {m.relevance}\n"
            f"Importance: {importance}"
        )

    def _embed_texts(self, texts: list[str]) -> np.ndarray:
        response = self.client.embeddings.create(model=self.embed_model, input=texts)
        vectors = [item.embedding for item in response.data]
        return np.array(vectors, dtype=np.float32)

    def _build_index(self) -> None:
        if not self.memories:
            raise ValueError("No memories found to index.")
        texts = [self._memory_embedding_text(m) for m in self.memories]
        embeddings = self._embed_texts(texts)
        faiss.normalize_L2(embeddings)

        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)
        self._index = index

    def _memory_source_type(self, memory: Memory) -> str:
        if memory.source_type:
            return memory.source_type
        tag_set = {t.lower() for t in memory.tags}
        if "chat_session" in tag_set:
            return "chat_session"
        if "shock_session" in tag_set:
            return "shock_session"
        if "decision" in tag_set:
            return "decision"
        if "interaction" in tag_set:
            return "interaction"
        return "base"

    def _parse_iso_datetime(self, value: str | None) -> datetime | None:
        if not value:
            return None
        text = value.strip()
        if not text:
            return None
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(text)
        except ValueError:
            return None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)

    def _recency_score(self, memory: Memory) -> float:
        timestamp = self._parse_iso_datetime(memory.created_at)
        if timestamp is None:
            source_type = self._memory_source_type(memory)
            if source_type in {"chat_session", "shock_session", "interaction", "decision"}:
                return self.SESSION_RECENCY_FALLBACK
            return self.BASE_RECENCY_FALLBACK

        age_days = max(0.0, (datetime.now(timezone.utc) - timestamp).total_seconds() / 86400.0)
        decay = math.exp(-math.log(2) * (age_days / self.RECENCY_HALF_LIFE_DAYS))
        return float(max(0.0, min(1.0, decay)))

    def _importance_score(self, memory: Memory) -> float:
        if memory.importance is not None:
            return float(max(0.0, min(1.0, memory.importance)))
        fallback = (0.60 * memory.relevance) + (0.40 * memory.intensity)
        return float(max(0.0, min(1.0, fallback)))

    def _normalise_similarity(self, similarity: float) -> float:
        return float(max(0.0, min(1.0, (similarity + 1.0) / 2.0)))

    def _hybrid_score(self, memory: Memory, similarity: float) -> float:
        semantic = self._normalise_similarity(similarity)
        recency = self._recency_score(memory)
        importance = self._importance_score(memory)
        return (
            (self.SEMANTIC_WEIGHT * semantic)
            + (self.RECENCY_WEIGHT * recency)
            + (self.IMPORTANCE_WEIGHT * importance)
        )

    def _scored_candidates(self, query: str, candidate_k: int | None = None) -> list[tuple[Memory, float]]:
        if self._index is None:
            raise RuntimeError("Memory index is not initialized.")

        q = self._embed_texts([query])
        faiss.normalize_L2(q)
        top_k = len(self.memories) if candidate_k is None else max(1, min(candidate_k, len(self.memories)))
        scores, indices = self._index.search(q, top_k)

        results: list[tuple[Memory, float]] = []
        for rank, idx in enumerate(indices[0]):
            if idx < 0 or idx >= len(self.memories):
                continue
            memory = self.memories[idx]
            similarity = float(scores[0][rank])
            results.append((memory, self._hybrid_score(memory, similarity)))
        results.sort(key=lambda item: item[1], reverse=True)
        return results

    def retrieve(self, query: str, k: int = 3) -> list[tuple[Memory, float]]:
        top_k = min(k, len(self.memories))
        candidate_k = min(max(top_k * 4, top_k), len(self.memories))
        return self._scored_candidates(query, candidate_k=candidate_k)[:top_k]

    def retrieve_with_source_quotas(
        self,
        query: str,
        *,
        base_k: int = 2,
        chat_session_k: int = 2,
        shock_session_k: int = 2,
        excluded_source_types: set[str] | None = None,
    ) -> list[tuple[Memory, float]]:
        excluded = {s.strip().lower() for s in (excluded_source_types or set()) if s and s.strip()}
        total_target = max(1, base_k + chat_session_k + shock_session_k)
        candidate_k = min(max(total_target * 6, total_target), len(self.memories))
        scored = self._scored_candidates(query, candidate_k=candidate_k)

        selected: list[tuple[Memory, float]] = []
        selected_ids: set[str] = set()

        def take(source_types: set[str], count: int) -> None:
            if count <= 0:
                return
            remaining = count
            for memory, score in scored:
                if remaining <= 0:
                    break
                if memory.id in selected_ids:
                    continue
                memory_source = self._memory_source_type(memory)
                if memory_source in excluded:
                    continue
                if memory_source not in source_types:
                    continue
                selected.append((memory, score))
                selected_ids.add(memory.id)
                remaining -= 1

        take({"base", "interaction", "decision"}, base_k)
        take({"chat_session"}, chat_session_k)
        take({"shock_session"}, shock_session_k)

        for memory, score in scored:
            if len(selected) >= total_target:
                break
            if memory.id in selected_ids:
                continue
            if self._memory_source_type(memory) in excluded:
                continue
            selected.append((memory, score))
            selected_ids.add(memory.id)

        selected.sort(key=lambda item: item[1], reverse=True)
        return selected[:total_target]

    def add_memory(self, memory: Memory, persist: bool = True) -> None:
        if self._index is None:
            raise RuntimeError("Memory index is not initialized.")

        emb = self._embed_texts([self._memory_embedding_text(memory)])
        faiss.normalize_L2(emb)

        self.memories.append(memory)
        self._index.add(emb)

        if persist and self.source_path is not None:
            self.source_path.parent.mkdir(parents=True, exist_ok=True)
            with self.source_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(memory.model_dump(), ensure_ascii=True) + "\n")
