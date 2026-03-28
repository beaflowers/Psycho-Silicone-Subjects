from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np

from .models import Memory


class MemoryStore:
    SEMANTIC_WEIGHT = 0.70
    INTENSITY_WEIGHT = 0.15
    RELEVANCE_WEIGHT = 0.15

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
    def from_jsonl(cls, path: Path, client, embed_model: str) -> "MemoryStore":
        memories: list[Memory] = []
        for raw in path.read_text(encoding="utf-8").splitlines():
            if not raw.strip():
                continue
            memories.append(Memory.model_validate(json.loads(raw)))
        return cls(client=client, memories=memories, embed_model=embed_model, source_path=path)

    def _memory_embedding_text(self, m: Memory) -> str:
        tags = ", ".join(m.tags)
        return (
            f"Memory: {m.text}\n"
            f"Tags: {tags}\n"
            f"Valence: {m.valence}\n"
            f"Intensity: {m.intensity}\n"
            f"Relevance: {m.relevance}"
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

    def retrieve(self, query: str, k: int = 3) -> list[tuple[Memory, float]]:
        if self._index is None:
            raise RuntimeError("Memory index is not initialized.")

        q = self._embed_texts([query])
        faiss.normalize_L2(q)
        top_k = min(k, len(self.memories))
        candidate_k = min(max(top_k * 2, top_k), len(self.memories))
        scores, indices = self._index.search(q, candidate_k)

        results: list[tuple[Memory, float]] = []
        for rank, idx in enumerate(indices[0]):
            memory = self.memories[idx]
            similarity = float(scores[0][rank])
            hybrid_score = (
                (self.SEMANTIC_WEIGHT * similarity)
                + (self.INTENSITY_WEIGHT * memory.intensity)
                + (self.RELEVANCE_WEIGHT * memory.relevance)
            )
            results.append((memory, hybrid_score))

        results.sort(key=lambda item: item[1], reverse=True)
        return results[:top_k]

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
