from enum import StrEnum

from pydantic import BaseModel


class PairChoice(StrEnum):
    A = "A"
    B = "B"


class PairVerdict(BaseModel):
    winner: PairChoice
    reason: str


class PairOutcome(BaseModel):
    """One pair, judged in both orders to cancel position bias."""

    first_promo_id: str
    second_promo_id: str
    first_score: float  # 1 = first won both orders, 0 = second won both, 0.5 = the orders disagreed (tie)
    reason_first_shown_first: str
    reason_second_shown_first: str


class PairwiseResult(BaseModel):
    outcomes: list[PairOutcome]
    strength_by_promo_id: dict[str, float]  # Bradley-Terry strength scaled to 0..1 within the pool
    failed_pairs: int
