from enum import StrEnum

from pydantic import BaseModel, Field


class SpoilerLevel(StrEnum):
    NONE = "NONE"
    MINOR = "MINOR"
    MAJOR = "MAJOR"


class ScriptLabel(BaseModel):
    """A speaker label as written in the script, valid for a range of episodes."""

    label: str
    first_episode: int
    last_episode: int


class CharacterEntry(BaseModel):
    canonical_name: str
    script_labels: list[ScriptLabel]
    role: str
    relationships: str
    secret: str = Field(description="What the audience learns later, or an empty string")


class ArcEntry(BaseModel):
    arc_id: str = Field(description="Short id like A1")
    name: str
    first_episode: int
    last_episode: int
    summary: str


class TwistEntry(BaseModel):
    description: str
    episode_number: int
    spoiler_level: SpoilerLevel


class EpisodeSummaryEntry(BaseModel):
    episode_number: int
    summary: str


class StoryBible(BaseModel):
    premise: str
    characters: list[CharacterEntry]
    arcs: list[ArcEntry]
    twists: list[TwistEntry]
    episode_summaries: list[EpisodeSummaryEntry]
    is_fallback: bool = False  # built from loglines only because the LLM call failed
