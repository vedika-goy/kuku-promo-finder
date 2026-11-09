from enum import StrEnum

from pydantic import BaseModel, Field

from promo_finder.screenplay.schemas import BlockKind
from promo_finder.tagging.schemas import CliffhangerTag, CliffhangerType, HookTag, HookType, LineImportance


class HookEvidence(StrEnum):
    CORROBORATED = "CORROBORATED"  # the LLM tagged it and the emotion or sound signal fires too
    CONTEXT_HOOK = "CONTEXT_HOOK"  # the LLM tagged it, no signal: typical of quiet reveals; must pass the cold viewer
    SIGNAL_ONLY = "SIGNAL_ONLY"  # the signal fired, the LLM missed it, and the review confirmed it


class ScoredLine(BaseModel):
    """One row of the merged per-line map (line_map.json)."""

    block_id: str
    episode_number: int
    kind: BlockKind
    start_s: float
    end_s: float
    text: str
    speaker: str | None
    intensity: float | None
    z_in_episode: float | None
    signal_strength: float  # strongest emotion spike or sound cue around the line, 0..1
    hook_score: float
    cliffhanger_score: float
    hook_evidence: HookEvidence | None
    hook_tag: HookTag | None
    cliffhanger_tag: CliffhangerTag | None
    arc_id: str | None
    importance: LineImportance | None
    is_recap: bool


class DisputedVerdict(StrEnum):
    HOOK = "HOOK"
    CLIFFHANGER = "CLIFFHANGER"
    NEITHER = "NEITHER"


class DisputedLineReview(BaseModel):
    block_id: str
    verdict: DisputedVerdict
    strength: int = Field(ge=1, le=5)
    hook_type: HookType | None
    cliffhanger_type: CliffhangerType | None
    open_question: str = Field(description="For a cliffhanger: the question the viewer is left with; else empty")


class DisputedReviewReply(BaseModel):
    reviews: list[DisputedLineReview]
