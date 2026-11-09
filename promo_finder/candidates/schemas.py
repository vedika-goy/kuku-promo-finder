from enum import StrEnum

from pydantic import BaseModel, Field

from promo_finder.signal_merge.schemas import HookEvidence
from promo_finder.story_bible.schemas import SpoilerLevel


class CandidateKind(StrEnum):
    SINGLE = "SINGLE"  # one continuous stretch
    COMPRESSED = "COMPRESSED"  # one episode, filler shots cut out
    CROSS_EPISODE = "CROSS_EPISODE"  # end of episode N + start of N+1, recap skipped


class RejectionReason(StrEnum):
    TOO_SHORT = "TOO_SHORT"
    TOO_LONG = "TOO_LONG"
    HOOK_LATE = "HOOK_LATE"
    ANSWER_INSIDE = "ANSWER_INSIDE"
    DIALOGUE_AFTER_CLIFFHANGER = "DIALOGUE_AFTER_CLIFFHANGER"
    SEGMENT_TOO_SHORT = "SEGMENT_TOO_SHORT"
    TOO_MANY_SEGMENTS = "TOO_MANY_SEGMENTS"
    CUT_THROUGH_LINE = "CUT_THROUGH_LINE"
    RECAP_INSIDE = "RECAP_INSIDE"
    COMPRESSION_FAILED = "COMPRESSION_FAILED"
    DUPLICATE = "DUPLICATE"
    COLD_VIEWER_CONFUSED = "COLD_VIEWER_CONFUSED"
    QUESTION_ANSWERED_INSIDE = "QUESTION_ANSWERED_INSIDE"
    JUDGE_FAILED = "JUDGE_FAILED"


class Segment(BaseModel):
    episode_number: int
    start_s: float
    end_s: float

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


class Candidate(BaseModel):
    candidate_id: str
    kind: CandidateKind
    segments: list[Segment]
    hook_block_id: str
    hook_start_block_id: str  # where the hook's sentence starts; earlier than hook_block_id when the hook line opens with "..."
    cliffhanger_block_id: str
    content_block_ids: list[str]  # dialogue/action/sound blocks that start inside the segments
    dropped_block_ids: list[str] = Field(default_factory=list)  # removed by compression; used by the repair step
    duration_s: float
    hook_offset_s: float
    hook_score: float
    cliffhanger_score: float
    hook_evidence: HookEvidence | None
    shape_score: float = 0.0
    duration_fit: float = 0.0
    early_episode_score: float = 0.0
    prefilter_score: float = 0.0
    spoiler_level: SpoilerLevel = SpoilerLevel.NONE
    needs_context: list[str] = Field(default_factory=list)
    open_question: str = ""
    hook_type: str | None = None
    arc_id: str | None = None
    flags: list[str] = Field(default_factory=list)
    rejection_reasons: list[RejectionReason] = Field(default_factory=list)


class CandidateSet(BaseModel):
    kept: list[Candidate]
    held_for_spoilers: list[Candidate]
    rejected: list[Candidate]
