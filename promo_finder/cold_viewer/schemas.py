from pydantic import BaseModel, Field

from promo_finder.candidates.schemas import Candidate


class OpeningReaction(BaseModel):
    """How strongly the first 3 seconds alone stop a scrolling stranger (judged without seeing the rest)."""

    stop_strength: int = Field(ge=1, le=5, description="1 = scroll past, 3 = pause briefly, 5 = can't look away")
    reason: str


class ViewerReaction(BaseModel):
    """What a stranger who has never seen the show makes of the whole clip."""

    what_is_happening: str
    who_is_who: str
    open_question: str
    confusing_parts: list[str]
    want_next_score: int = Field(ge=1, le=5)


class GraderVerdict(BaseModel):
    comprehension: int = Field(ge=0, le=2, description="2 = relationships and stakes right, 1 = partly, 0 = wrong or lost")
    cliffhanger_match: int = Field(ge=0, le=2, description="2 = the viewer's question is the intended one, 1 = a related real question, 0 = none")
    is_question_answered_inside: bool = Field(description="True when the clip itself resolves the tension it ends on")
    misreadings: list[str]
    note: str


class JudgedCandidate(BaseModel):
    candidate: Candidate
    opening_reaction: OpeningReaction | None = None
    viewer_reaction: ViewerReaction | None = None
    grader_verdict: GraderVerdict | None = None
    is_repaired: bool = False
    rendered_clip: str = ""
