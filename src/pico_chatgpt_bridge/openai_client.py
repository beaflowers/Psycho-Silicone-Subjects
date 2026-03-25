"""Minimal OpenAI client setup."""

from __future__ import annotations

import os

from openai import OpenAI
from dotenv import load_dotenv


load_dotenv()
MODEL_NAME = os.environ.get("OPENAI_CHAT_MODEL", "gpt-5-mini")


def create_client() -> OpenAI:
    """Create an OpenAI client using OPENAI_API_KEY from env or .env."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set in the environment or .env.")
    return OpenAI(api_key=api_key)


def get_model_name() -> str:
    """Return the model name used by this project."""
    return MODEL_NAME
