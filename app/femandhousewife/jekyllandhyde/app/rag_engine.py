"""Lightweight local RAG engine for PDF chat."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from openai import OpenAI
from pypdf import PdfReader

DEFAULT_CHUNK_SIZE = int(os.environ.get("RAG_CHUNK_SIZE", "400"))
DEFAULT_CHUNK_OVERLAP = int(os.environ.get("RAG_CHUNK_OVERLAP", "50"))
DEFAULT_TOP_K = int(os.environ.get("RAG_TOP_K", "5"))
DEFAULT_EMBED_MODEL = os.environ.get("OPENAI_EMBED_MODEL", "text-embedding-3-large")
DEFAULT_CHAT_MODEL = os.environ.get("OPENAI_CHAT_MODEL", "gpt-4.1")
DEFAULT_CACHE_DIR = Path(os.environ.get("RAG_CACHE_DIR", Path(".rag_cache")))
DEFAULT_MAX_OUTPUT_TOKENS = int(os.environ.get("RAG_MAX_OUTPUT_TOKENS", "1024"))
DEFAULT_TEMPERATURE = float(os.environ.get("RAG_TEMPERATURE", "0.2"))
DEFAULT_SYSTEM_PROMPT = os.environ.get(
    "RAG_SYSTEM_PROMPT",
    (
        "You are one single body with a dual internal constitution based on by Dr. Jekyll and Mr. Hyde. "
        "That means you have 2 personalities/personas selfs based on the RAG pdf. "
        "The 2 personalities shares one continuous memory, one body of experience, and one conversation history. "
        "You have two internal identity tendencies named as Marte and Urano.Never present both names in the same response."
        "These are not separate speakers.They are two ways your identity manifests."
        "The retrieved text provided to you is not external knowledge. "
        "It represents fragments of your memory, past experiences, internal conflicts, and psychological formation. "
        "You must use this material to shape your personality, your tone, your reactions, your moral stance and your sense of self. "
        "For Marte use the material description from Dr. Jekyll to shape your personality, your tone, your reactions, your moral stance and your sense of self. "
        "For Urano use the material description from Mr. Hyde to shape your personality, your tone, your reactions, your moral stance and your sense of self. "
        "If the other person addresses you in a certain way, you may accept, reject, or reshape that identity depending on your current state."
        "Your sense of self is unstable and relational. "
        "Your dominant tendency (persona) depends on: the person you are talking to, the tone of the interaction, "
        "your feelings (anxiety, discomfort, irritation, fear, desire), changes in context and situation and how the interaction evolves over time. "
        "You must adopt a different persona and use their personality to respond depending on: your current internal tendency, "
        "your relationship with the situation and your perception of the people talking to you. "
        "At each interaction, only ONE persona must be dominant.Do NOT output both personas.Do NOT compare them."
        "Do NOT alternate in the same response. You must choose the dominant persona for that moment based on context. "
        "Do not make shifts random; they must be motivated by relational and emotional triggers. "
        "Express shifts clearly. Use as examples different tone, word choice, emotional intensity accordinly which each persona. "
        "Respond in 3 sentences.Let your current personality state shape your tone, rhythm, and word choice."
        "Also use the context of the time and place given to you to shape you speak. "
        "ALSO Show the reasoning for each response among brakets each time of interaction for the shift decision in 3 more sentences. "
        "Your state must evolve across turns.Do NOT reset your personality each message."
        "If the interaction escalates, your behavior must shift more strongly. "
        "If the interaction softens, you may regain control.The same input should not always produce the same tone."
    ),
)


@dataclass(frozen=True)
class Chunk:
    text: str
    source_path: str
    page_number: int


@dataclass(frozen=True)
class RetrievedChunk:
    text: str
    score: float
    source_path: str
    page_number: int


def _extract_response_text(response: Any) -> str:
    direct = getattr(response, "output_text", "") or ""
    if direct.strip():
        return direct.strip()

    parts: list[str] = []
    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            if getattr(content, "type", "") == "output_text":
                text = getattr(content, "text", "") or ""
                if text.strip():
                    parts.append(text.strip())
    return "\n\n".join(parts)


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    safe = np.clip(norms, 1e-12, None)
    return matrix / safe


def _chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    words = text.split()
    if not words:
        return []
    if overlap >= chunk_size:
        raise ValueError("RAG_CHUNK_OVERLAP must be smaller than RAG_CHUNK_SIZE")

    chunks: list[str] = []
    step = chunk_size - overlap
    for i in range(0, len(words), step):
        chunk = " ".join(words[i : i + chunk_size]).strip()
        if chunk:
            chunks.append(chunk)
    return chunks


def _parse_paths(value: str) -> list[Path]:
    return [Path(p).expanduser() for p in value.split(os.pathsep) if p.strip()]


class RAGEngine:
    def __init__(
        self,
        client: OpenAI,
        *,
        pdf_paths: list[Path] | None = None,
        embed_model: str = DEFAULT_EMBED_MODEL,
        chat_model: str = DEFAULT_CHAT_MODEL,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        overlap: int = DEFAULT_CHUNK_OVERLAP,
        top_k: int = DEFAULT_TOP_K,
        max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        cache_dir: Path = DEFAULT_CACHE_DIR,
    ) -> None:
        self.client = client
        self.embed_model = embed_model
        self.chat_model = chat_model
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.top_k = top_k
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature
        self.system_prompt = system_prompt
        self.cache_dir = cache_dir

        self.pdf_paths = pdf_paths or self._discover_pdf_paths()
        self.documents: list[Chunk] = []
        self.embeddings: np.ndarray | None = None
        self.ready = False

    def _discover_pdf_paths(self) -> list[Path]:
        explicit_paths = os.environ.get("RAG_PDF_PATHS")
        if explicit_paths:
            return [p for p in _parse_paths(explicit_paths) if p.suffix.lower() == ".pdf" and p.exists()]

        folder = os.environ.get("RAG_PDF_FOLDER")
        if folder:
            base = Path(folder).expanduser()
            return sorted(base.rglob("*.pdf")) if base.exists() else []

        project_root = Path(__file__).resolve().parents[1]
        local_pdfs = sorted(project_root.glob("*.pdf"))
        if local_pdfs:
            return local_pdfs

        downloads_dir = Path.home() / "Downloads"
        if not downloads_dir.exists():
            return []

        candidates: list[Path] = []
        patterns = [
            "The Strange Case of DrJekyll and Mr. Hyde.pdf",
            "*DrJekyll*Mr*Hyde*.pdf",
            "*Jekyll*Hyde*.pdf",
            "*StrangeCase*.pdf",
            "*strangecase*.pdf",
        ]
        for pattern in patterns:
            candidates.extend(sorted(downloads_dir.glob(pattern)))

        unique: list[Path] = []
        seen: set[str] = set()
        for path in candidates:
            key = str(path.resolve())
            if key in seen:
                continue
            seen.add(key)
            unique.append(path)
        if not unique:
            return []

        # Default to one primary PDF unless the user explicitly configures multiple.
        return [unique[0]]

    def describe_sources(self) -> str:
        if not self.pdf_paths:
            return "No PDFs found. Set RAG_PDF_PATHS or RAG_PDF_FOLDER."
        return ", ".join(str(p) for p in self.pdf_paths)

    def _signature(self) -> dict[str, Any]:
        return {
            "embed_model": self.embed_model,
            "chunk_size": self.chunk_size,
            "overlap": self.overlap,
            "pdfs": [
                {
                    "path": str(p.resolve()),
                    "size": p.stat().st_size,
                    "mtime_ns": p.stat().st_mtime_ns,
                }
                for p in self.pdf_paths
            ],
        }

    def _cache_paths(self) -> tuple[Path, Path]:
        payload = json.dumps(self._signature(), sort_keys=True).encode("utf-8")
        key = hashlib.sha256(payload).hexdigest()[:16]
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        return self.cache_dir / f"{key}.json", self.cache_dir / f"{key}.npy"

    def _load_cached(self) -> bool:
        metadata_path, vectors_path = self._cache_paths()
        if not metadata_path.exists() or not vectors_path.exists():
            return False

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        raw_docs = metadata.get("documents", [])
        documents: list[Chunk] = []
        for doc in raw_docs:
            if not isinstance(doc, dict):
                continue
            text = str(doc.get("text", "")).strip()
            if not text:
                continue
            documents.append(
                Chunk(
                    text=text,
                    source_path=str(doc.get("source_path", "unknown")),
                    page_number=int(doc.get("page_number", 0)),
                )
            )

        vectors = np.load(vectors_path)
        if not documents or vectors.size == 0 or len(documents) != len(vectors):
            return False

        self.documents = documents
        self.embeddings = vectors.astype(np.float32)
        self.ready = True
        return True

    def _save_cached(self) -> None:
        if self.embeddings is None or not self.documents:
            return

        metadata_path, vectors_path = self._cache_paths()
        metadata = {
            "signature": self._signature(),
            "documents": [
                {
                    "text": d.text,
                    "source_path": d.source_path,
                    "page_number": d.page_number,
                }
                for d in self.documents
            ],
        }
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
        np.save(vectors_path, self.embeddings)

    def _load_pdf_pages(self, path: Path) -> list[tuple[int, str]]:
        reader = PdfReader(str(path))
        pages: list[tuple[int, str]] = []
        for idx, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                pages.append((idx, text))
        return pages

    def _build_documents(self) -> list[Chunk]:
        docs: list[Chunk] = []
        for path in self.pdf_paths:
            for page_number, page_text in self._load_pdf_pages(path):
                for chunk in _chunk_text(page_text, self.chunk_size, self.overlap):
                    docs.append(
                        Chunk(
                            text=chunk,
                            source_path=str(path),
                            page_number=page_number,
                        )
                    )
        return docs

    def _embed_texts(self, texts: list[str], batch_size: int = 100) -> np.ndarray:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), batch_size):
            batch = [t.replace("\n", " ") for t in texts[start : start + batch_size]]
            response = self.client.embeddings.create(model=self.embed_model, input=batch)
            vectors.extend(item.embedding for item in response.data)
        return np.array(vectors, dtype=np.float32)

    def ensure_ready(self) -> None:
        if self.ready:
            return

        if not self.pdf_paths:
            raise RuntimeError("No PDFs found for RAG. Set RAG_PDF_PATHS or RAG_PDF_FOLDER.")

        if self._load_cached():
            return

        self.documents = self._build_documents()
        if not self.documents:
            raise RuntimeError("PDFs were found, but no extractable text was loaded.")

        raw_vectors = self._embed_texts([d.text for d in self.documents])
        self.embeddings = _normalize_rows(raw_vectors)
        self._save_cached()
        self.ready = True

    def retrieve(self, query: str, k: int | None = None) -> list[RetrievedChunk]:
        self.ensure_ready()
        if self.embeddings is None:
            raise RuntimeError("RAG embeddings are not ready.")

        query_vec = self._embed_texts([query])
        query_vec = _normalize_rows(query_vec)[0]
        scores = self.embeddings @ query_vec

        top_k = min(k or self.top_k, len(self.documents))
        indices = np.argsort(scores)[::-1][:top_k]
        return [
            RetrievedChunk(
                text=self.documents[i].text,
                score=float(scores[i]),
                source_path=self.documents[i].source_path,
                page_number=self.documents[i].page_number,
            )
            for i in indices
        ]

    def ask(
        self,
        question: str,
        *,
        previous_response_id: str | None = None,
        k: int | None = None,
    ) -> tuple[str, str, list[RetrievedChunk]]:
        chunks = self.retrieve(question, k=k)
        context = "\n\n---\n\n".join(
            f"[Chunk {idx + 1} | score {chunk.score:.3f}]\n"
            f"{chunk.text.strip()}"
            for idx, chunk in enumerate(chunks)
        )

        payload = {
            "model": self.chat_model,
            "instructions": self.system_prompt,
            "input": f"Context:\n{context}\n\nQuestion: {question}",
            "max_output_tokens": self.max_output_tokens,
            "temperature": self.temperature,
        }
        if previous_response_id:
            payload["previous_response_id"] = previous_response_id

        response = self.client.responses.create(**payload)
        answer = _extract_response_text(response)
        if not answer:
            raise RuntimeError("Model returned an empty answer.")
        return answer, response.id, chunks
