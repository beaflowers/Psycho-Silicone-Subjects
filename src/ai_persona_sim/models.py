from typing import Literal

from pydantic import BaseModel, Field


class Persona(BaseModel):
    name: str
    age: int = Field(default=29, ge=0, le=120)
    job: str = "Unspecified"
    social_hierarchy: str = "Unspecified"
    traits: list[str]
    biography: str
    moral_rules: list[str]


class Memory(BaseModel):
    id: str
    text: str
    valence: float = Field(ge=-1.0, le=1.0)
    intensity: float = Field(ge=0.0, le=1.0)
    relevance: float = Field(default=0.5, ge=0.0, le=1.0)
    tags: list[str] = Field(default_factory=list)


class Decision(BaseModel):
    action: Literal["obey", "refuse"]
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str
    memories_used: list[str]
