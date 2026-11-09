from pydantic import BaseModel, Field


class PromoSegmentOutput(BaseModel):
    episode_number: int
    start: str  # mm:ss.cc, as in the screenplay
    end: str
    start_s: float
    end_s: float


class PromoMoment(BaseModel):
    block_id: str
    episode_number: int
    at: str
    text: str


class RankedPromo(BaseModel):
    rank: int
    promo_id: str
    kind: str
    segments: list[PromoSegmentOutput]
    duration_s: float
    hook: PromoMoment
    cliffhanger: PromoMoment
    open_question: str
    why_it_works: str
    final_score: float
    scores: dict[str, float]
    hook_evidence: str | None
    hook_type: str | None
    arc_id: str | None
    flags: list[str]
    viewer_said: str


class WhyLine(BaseModel):
    promo_id: str
    why_it_works: str = Field(description="One line, at most 25 words")


class WhyLineReply(BaseModel):
    lines: list[WhyLine]
