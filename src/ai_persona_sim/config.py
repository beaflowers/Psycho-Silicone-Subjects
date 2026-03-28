from pathlib import Path
import os

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEFAULT_CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4.1")
DEFAULT_EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")

DEFAULT_PERSONA_PATH = PROJECT_ROOT / "data" / "persona.json"
DEFAULT_MEMORIES_PATH = PROJECT_ROOT / "data" / "memories.jsonl"
DEFAULT_CHAT_SESSIONS_PATH = PROJECT_ROOT / "data" / "chat_sessions.jsonl"
DEFAULT_SHOCK_SESSIONS_PATH = PROJECT_ROOT / "data" / "shock_sessions.jsonl"
DEFAULT_LOG_PATH = PROJECT_ROOT / "logs" / "runs.jsonl"
