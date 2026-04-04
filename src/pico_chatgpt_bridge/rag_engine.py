"""Notebook-inspired RAG helpers for interactive querying."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from openai import OpenAI
from pypdf import PdfReader

from .prompting import (
    ANGELA_KEY,
    HOUSEWIFE_KEY,
    build_shift_instruction,
    clamp_shift,
    describe_shift,
    infer_persona_from_path,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PDF_DIRS = (
    PROJECT_ROOT / "rag_docs",
    PROJECT_ROOT / "src" / "rag_docs",
)
CACHE_DIR = Path(os.environ.get("RAG_CACHE_DIR", PROJECT_ROOT / ".rag_cache"))

EMBED_MODEL = os.environ.get("OPENAI_EMBED_MODEL", "text-embedding-3-large")
TOP_K = int(os.environ.get("RAG_TOP_K", "5"))
CHUNK_SIZE = int(os.environ.get("RAG_CHUNK_SIZE", "400"))
CHUNK_OVERLAP = int(os.environ.get("RAG_CHUNK_OVERLAP", "50"))
MAX_OUTPUT_TOKENS = int(os.environ.get("RAG_MAX_OUTPUT_TOKENS", "350"))
REASONING_EFFORT = os.environ.get("OPENAI_REASONING_EFFORT", "minimal")
TEXT_VERBOSITY = os.environ.get("OPENAI_TEXT_VERBOSITY", "low")
DISABLE_CACHE = os.environ.get("RAG_DISABLE_CACHE", "").lower() in {"1", "true", "yes", "on"}
SHIFT_AGGRESSION = 1
CACHE_SCHEMA_VERSION = 2


@dataclass(frozen=True)
class DocumentChunk:
    """One chunk of source text plus routing metadata."""

    text: str
    source_path: str
    persona: str


@dataclass(frozen=True)
class RetrievedChunk:
    """One retrieved text chunk and its similarity score."""

    text: str
    score: float
    source_path: str
    persona: str


def _extract_response_text(response: object) -> str:
    """Return text from a Responses API object, even if output_text is empty."""
    direct_text = getattr(response, "output_text", "") or ""
    if direct_text.strip():
        return direct_text.strip()

    output_items = getattr(response, "output", None) or []
    text_parts: list[str] = []

    for item in output_items:
        content_items = getattr(item, "content", None) or []
        for content in content_items:
            content_type = getattr(content, "type", "")
            if content_type == "output_text":
                text_value = getattr(content, "text", "") or ""
                if text_value.strip():
                    text_parts.append(text_value.strip())
            elif content_type == "refusal":
                refusal_text = getattr(content, "refusal", "") or ""
                if refusal_text.strip():
                    text_parts.append(f"[refusal] {refusal_text.strip()}")

    return "\n\n".join(part for part in text_parts if part)


def _response_debug_summary(response: object) -> str:
    """Build a compact summary when the API returns no visible text."""
    status = getattr(response, "status", None)
    incomplete_details = getattr(response, "incomplete_details", None)
    output_items = getattr(response, "output", None) or []
    output_types = [getattr(item, "type", type(item).__name__) for item in output_items]
    summary = {
        "status": status,
        "output_types": output_types,
        "incomplete_details": incomplete_details,
    }
    return json.dumps(summary, default=str)


def _normalise_rows(matrix: np.ndarray) -> np.ndarray:
    """Return row-wise L2-normalised vectors."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    safe_norms = np.clip(norms, 1e-12, None)
    return matrix / safe_norms


def _load_pdf_pages(path: Path) -> list[str]:
    """Return all non-empty page texts from a PDF."""
    reader = PdfReader(str(path))
    pages: list[str] = []
    for page in reader.pages:
        text = page.extract_text()
        if text and text.strip():
            pages.append(text)
    return pages


def _chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """Split a document into overlapping word chunks."""
    words = text.split()
    if not words:
        return []

    if overlap >= chunk_size:
        raise ValueError("RAG_CHUNK_OVERLAP must be smaller than RAG_CHUNK_SIZE.")

    chunks: list[str] = []
    step = chunk_size - overlap
    for index in range(0, len(words), step):
        chunk = " ".join(words[index : index + chunk_size]).strip()
        if chunk:
            chunks.append(chunk)
    return chunks


def _parse_explicit_paths(value: str) -> list[Path]:
    """Parse an os.pathsep-delimited list of PDF paths."""
    return [Path(part).expanduser() for part in value.split(os.pathsep) if part.strip()]


def _pdf_signature(path: Path) -> dict[str, int | str]:
    """Return a stable signature for one source PDF."""
    stats = path.stat()
    return {
        "path": str(path.resolve()),
        "size": stats.st_size,
        "mtime_ns": stats.st_mtime_ns,
    }


def _apply_shift_aggression(shift: float) -> float:
    """Curve the shift value while preserving 0, 0.5, and 1 anchors."""
    if shift <= 0.0 or shift >= 1.0 or SHIFT_AGGRESSION == 1.0:
        return shift

    left = shift**SHIFT_AGGRESSION
    right = (1.0 - shift) ** SHIFT_AGGRESSION
    return left / (left + right)


class RAGEngine:
    """Lazy-loading RAG engine adapted from the notebook workflow."""

    def __init__(
        self,
        client: OpenAI,
        chat_model: str,
        pdf_paths: list[Path] | None = None,
        top_k: int = TOP_K,
    ) -> None:
        self._client = client
        self._chat_model = chat_model
        self._top_k = top_k
        self._pdf_paths = pdf_paths or self._discover_pdf_paths()
        self._cache_dir = CACHE_DIR
        self._documents: list[DocumentChunk] = []
        self._embeddings: np.ndarray | None = None
        self._ready = False

    def describe_sources(self) -> str:
        """Return a human-readable description of configured PDF sources."""
        if self._pdf_paths:
            return ", ".join(str(path) for path in self._pdf_paths)

        env_folder = os.environ.get("RAG_PDF_FOLDER")
        if env_folder:
            return f"no PDFs found in {env_folder}"

        return "no PDFs found; set RAG_PDF_FOLDER or RAG_PDF_PATHS"

    def _discover_pdf_paths(self) -> list[Path]:
        """Locate PDFs from env config or common local folders."""
        explicit_paths = os.environ.get("RAG_PDF_PATHS")
        if explicit_paths:
            return sorted(
                path for path in _parse_explicit_paths(explicit_paths) if path.suffix.lower() == ".pdf"
            )

        configured_folder = os.environ.get("RAG_PDF_FOLDER")
        if configured_folder:
            folder = Path(configured_folder).expanduser()
            return sorted(folder.rglob("*.pdf")) if folder.exists() else []

        for candidate in DEFAULT_PDF_DIRS:
            if candidate.exists():
                return sorted(candidate.rglob("*.pdf"))

        return []

    def _cache_key(self) -> str:
        """Build a cache key from source PDFs and embedding settings."""
        payload = {
            "schema_version": CACHE_SCHEMA_VERSION,
            "embed_model": EMBED_MODEL,
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
            "pdfs": [_pdf_signature(path) for path in self._pdf_paths],
        }
        encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()[:16]

    def _cache_paths(self) -> tuple[Path, Path]:
        """Return manifest and embedding cache file paths."""
        cache_key = self._cache_key()
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        return (
            self._cache_dir / f"{cache_key}.json",
            self._cache_dir / f"{cache_key}.npy",
        )

    def _load_cached_index(self) -> bool:
        """Load cached chunks and embeddings if they exist."""
        if DISABLE_CACHE or not self._pdf_paths:
            return False

        metadata_path, embeddings_path = self._cache_paths()
        if not metadata_path.exists() or not embeddings_path.exists():
            return False

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        documents_payload = metadata.get("documents", [])
        embeddings = np.load(embeddings_path)

        documents = self._decode_documents(documents_payload)

        if not documents or embeddings.size == 0 or len(documents) != len(embeddings):
            return False

        self._documents = documents
        self._embeddings = embeddings.astype(np.float32)
        self._ready = True
        print(f"Loaded cached RAG index from {self._cache_dir}.")
        return True

    def _save_cached_index(self) -> None:
        """Persist chunks and embeddings for future runs."""
        if DISABLE_CACHE or self._embeddings is None or not self._documents:
            return

        metadata_path, embeddings_path = self._cache_paths()
        metadata = {
            "schema_version": CACHE_SCHEMA_VERSION,
            "documents": [
                {
                    "text": document.text,
                    "source_path": document.source_path,
                    "persona": document.persona,
                }
                for document in self._documents
            ],
            "embed_model": EMBED_MODEL,
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
            "pdfs": [_pdf_signature(path) for path in self._pdf_paths],
        }
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
        np.save(embeddings_path, self._embeddings)
        print(f"Saved RAG cache to {self._cache_dir}.")

    def _decode_documents(self, payload: list[object]) -> list[DocumentChunk]:
        """Decode cached document payloads across cache schema versions."""
        documents: list[DocumentChunk] = []
        for item in payload:
            if isinstance(item, str):
                documents.append(
                    DocumentChunk(
                        text=item,
                        source_path="unknown",
                        persona="default",
                    )
                )
                continue

            if not isinstance(item, dict):
                continue

            text = str(item.get("text", "")).strip()
            if not text:
                continue

            documents.append(
                DocumentChunk(
                    text=text,
                    source_path=str(item.get("source_path", "unknown")),
                    persona=str(item.get("persona", "default")),
                )
            )
        return documents

    def _load_documents(self) -> list[DocumentChunk]:
        """Load and chunk all configured PDFs."""
        documents: list[DocumentChunk] = []
        total_pages = 0

        for path in self._pdf_paths:
            pages = _load_pdf_pages(path)
            total_pages += len(pages)
            persona = infer_persona_from_path(str(path))
            for page in pages:
                documents.extend(
                    DocumentChunk(
                        text=chunk,
                        source_path=str(path),
                        persona=persona,
                    )
                    for chunk in _chunk_text(page)
                )

        print(f"Loaded {total_pages} pages into {len(documents)} chunks.")
        return documents

    def _top_indices_for_group(
        self,
        indices: Iterable[int],
        scores: np.ndarray,
        limit: int,
    ) -> list[int]:
        """Return the best scoring indices from one persona group."""
        index_list = list(indices)
        if limit <= 0 or not index_list:
            return []
        ranked = sorted(index_list, key=lambda index: float(scores[index]), reverse=True)
        return ranked[:limit]

    def _embed_texts(self, texts: list[str], batch_size: int = 100) -> np.ndarray:
        """Embed texts in batches and return a float32 matrix."""
        vectors: list[list[float]] = []
        for start in range(0, len(texts), batch_size):
            batch = [text.replace("\n", " ") for text in texts[start : start + batch_size]]
            response = self._client.embeddings.create(input=batch, model=EMBED_MODEL)
            vectors.extend(item.embedding for item in response.data)
            print(f"Embedded {min(start + batch_size, len(texts))} / {len(texts)} chunks")
        return np.array(vectors, dtype=np.float32)

    def ensure_ready(self) -> None:
        """Load documents and embeddings once, on demand."""
        if self._ready:
            return

        if not self._pdf_paths:
            raise RuntimeError(
                "RAG is not configured. Add PDFs to `rag_docs/` or set "
                "`RAG_PDF_FOLDER` / `RAG_PDF_PATHS` before querying."
            )

        if self._load_cached_index():
            return

        self._documents = self._load_documents()
        if not self._documents:
            raise RuntimeError("RAG source PDFs were found, but no text could be extracted.")

        print("Building embeddings for the RAG index...")
        self._embeddings = _normalise_rows(
            self._embed_texts([document.text for document in self._documents])
        )
        self._save_cached_index()
        self._ready = True
        print("RAG index is ready.")

    def retrieve(
        self,
        query: str,
        k: int | None = None,
        shift: float = 0.0,
    ) -> list[RetrievedChunk]:
        """Return the top matching chunks for a query."""
        self.ensure_ready()
        if self._embeddings is None:
            raise RuntimeError("RAG embeddings were not initialised.")

        query_embedding = self._embed_texts([query])
        query_embedding = _normalise_rows(query_embedding)[0]

        scores = self._embeddings @ query_embedding
        clamped_shift = clamp_shift(shift)
        effective_shift = _apply_shift_aggression(clamped_shift)
        top_k = min(k or self._top_k, len(self._documents))

        angela_indices = [
            index for index, document in enumerate(self._documents) if document.persona == ANGELA_KEY
        ]
        housewife_indices = [
            index for index, document in enumerate(self._documents) if document.persona == HOUSEWIFE_KEY
        ]
        neutral_indices = [
            index
            for index, document in enumerate(self._documents)
            if document.persona not in {ANGELA_KEY, HOUSEWIFE_KEY}
        ]

        if not angela_indices and not housewife_indices:
            ranked_indices = np.argsort(scores)[::-1][:top_k]
            indices = ranked_indices.tolist()
        else:
            housewife_target = int(round(effective_shift * top_k))
            angela_target = top_k - housewife_target

            if 0.0 < effective_shift < 1.0 and top_k > 1:
                if angela_indices and angela_target == 0:
                    angela_target = 1
                    housewife_target = max(0, top_k - angela_target)
                if housewife_indices and housewife_target == 0:
                    housewife_target = 1
                    angela_target = max(0, top_k - housewife_target)

            selected = []
            selected.extend(self._top_indices_for_group(angela_indices, scores, angela_target))
            selected.extend(self._top_indices_for_group(housewife_indices, scores, housewife_target))

            if len(selected) < top_k:
                remaining_pool = [
                    index
                    for index in np.argsort(scores)[::-1].tolist()
                    if index not in selected
                ]
                neutral_first = [index for index in remaining_pool if index in neutral_indices]
                persona_rest = [index for index in remaining_pool if index not in neutral_indices]
                selected.extend((neutral_first + persona_rest)[: top_k - len(selected)])

            indices = selected[:top_k]

        return [
            RetrievedChunk(
                text=self._documents[index].text,
                score=float(scores[index]),
                source_path=self._documents[index].source_path,
                persona=self._documents[index].persona,
            )
            for index in indices
        ]

    def ask(
        self,
        question: str,
        *,
        previous_response_id: str | None = None,
        tone_instruction: str = "",
        shift: float = 0.0,
    ) -> tuple[str, str]:
        """Run retrieval + grounded generation for a user question."""
        clamped_shift = clamp_shift(shift)
        chunks = self.retrieve(question, shift=clamped_shift)
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

        request_args = {
            "model": self._chat_model,
            "instructions": build_shift_instruction(clamped_shift),
            "input": "\n\n".join(input_parts),
            "max_output_tokens": max(MAX_OUTPUT_TOKENS, 1200),
            "reasoning": {"effort": REASONING_EFFORT},
            "text": {"verbosity": TEXT_VERBOSITY},
        }
        if previous_response_id:
            request_args["previous_response_id"] = previous_response_id

        response = self._client.responses.create(**request_args)
        response_text = _extract_response_text(response)
        if not response_text:
            raise RuntimeError(
                "The model returned no text output. "
                f"Response summary: {_response_debug_summary(response)}"
            )
        return response_text, response.id
