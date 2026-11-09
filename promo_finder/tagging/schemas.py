from enum import StrEnum

from pydantic import BaseModel, Field

from promo_finder.story_bible.schemas import SpoilerLevel


class HookType(StrEnum):
    PHYSICAL = "PHYSICAL"
    REVEAL = "REVEAL"
    HUMILIATION = "HUMILIATION"
    POWER_REVERSAL = "POWER_REVERSAL"
    THREAT = "THREAT"
    SHOCK_LINE = "SHOCK_LINE"
    ROMANCE = "ROMANCE"


class CliffhangerType(StrEnum):
    CUT_BEFORE_ANSWER = "CUT_BEFORE_ANSWER"
    REVEAL_PENDING = "REVEAL_PENDING"
    THREAT_PENDING = "THREAT_PENDING"
    ARRIVAL = "ARRIVAL"
    INTERRUPTED_ACTION = "INTERRUPTED_ACTION"
    DECISION_PENDING = "DECISION_PENDING"


class LineImportance(StrEnum):
    ESSENTIAL = "ESSENTIAL"
    SUPPORTING = "SUPPORTING"


ANSWERED_LATER = "LATER"  # answered in a later episode
ANSWERED_NEVER = "NOT_IN_PACK"


class HookTag(BaseModel):
    block_id: str
    strength: int = Field(ge=1, le=5)
    hook_type: HookType
    trigger_block_id: str | None = Field(description="The setup line just before the hook, if the hook needs it; else null")
    needs_context: list[str] = Field(description="Facts a stranger must already know for the hook to land")
    spoiler_level: SpoilerLevel
    reason: str


class CliffhangerTag(BaseModel):
    block_id: str
    strength: int = Field(ge=1, le=5)
    cliffhanger_type: CliffhangerType
    open_question: str
    answered_at: str = Field(description="block_id of the line that answers the question, or LATER, or NOT_IN_PACK")


class ArcLineTag(BaseModel):
    block_id: str
    arc_id: str
    importance: LineImportance


class EpisodeTags(BaseModel):
    episode_number: int
    summary: str
    hooks: list[HookTag]
    cliffhangers: list[CliffhangerTag]
    arc_lines: list[ArcLineTag]
    is_untagged: bool = False  # the LLM call failed; only model-free signals exist for this episode
    validation_notes: list[str] = Field(default_factory=list)
