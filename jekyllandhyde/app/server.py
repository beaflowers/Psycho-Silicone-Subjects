"""FastAPI server for local PDF RAG chat."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from openai import OpenAI
from pydantic import BaseModel, Field

from .rag_engine import RAGEngine


load_dotenv()


def _require_api_key() -> str:
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key:
        raise RuntimeError("OPENAI_API_KEY is missing. Add it to .env or your shell environment.")
    return key


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    session_id: str = Field(default="default", min_length=1, max_length=200)
    top_k: int | None = Field(default=None, ge=1, le=20)


class ResetRequest(BaseModel):
    session_id: str = Field(default="default", min_length=1, max_length=200)


app = FastAPI(title="Dr Jekyll & Mr Hyde RAG Chat")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

project_root = Path(__file__).resolve().parents[1]
web_dir = project_root / "web"

conversation_state: dict[str, str] = {}
rag: RAGEngine | None = None

if web_dir.exists():
    app.mount("/static", StaticFiles(directory=web_dir), name="static")


@app.get("/")
def home() -> FileResponse:
    index = web_dir / "index.html"
    if not index.exists():
        raise HTTPException(status_code=404, detail="web/index.html not found")
    return FileResponse(index)


@app.get("/api/health")
def health() -> dict[str, Any]:
    key_present = bool(os.environ.get("OPENAI_API_KEY", "").strip())
    return {
        "ok": True,
        "ready": bool(rag and rag.ready),
        "key_configured": key_present,
        "sources": rag.describe_sources() if rag else "Not initialized yet",
    }


@app.post("/api/reset")
def reset(data: ResetRequest) -> dict[str, Any]:
    conversation_state.pop(data.session_id, None)
    return {"ok": True, "session_id": data.session_id}


@app.post("/api/chat")
def chat(data: ChatRequest) -> dict[str, Any]:
    try:
        active_rag = _get_rag()
        previous_response_id = conversation_state.get(data.session_id)
        answer, response_id, chunks = active_rag.ask(
            data.message,
            previous_response_id=previous_response_id,
            k=data.top_k,
        )
        conversation_state[data.session_id] = response_id
        return {
            "ok": True,
            "answer": answer,
            "session_id": data.session_id,
            "sources": [
                {
                    "path": c.source_path,
                    "page": c.page_number,
                    "score": round(c.score, 4),
                }
                for c in chunks
            ],
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _get_rag() -> RAGEngine:
    global rag
    if rag is not None:
        return rag

    key = _require_api_key()
    client = OpenAI(api_key=key)
    rag = RAGEngine(client=client)
    return rag
