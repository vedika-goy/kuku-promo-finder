"""Order the best promos by head-to-head judgments, which separate them better than absolute 1-5 scores."""

import logging
import math
from concurrent.futures import ThreadPoolExecutor

from promo_finder import config
from promo_finder.cold_viewer.schemas import JudgedCandidate
from promo_finder.cold_viewer.service import JudgeContext, render_clip
from promo_finder.pairwise_comparison import prompts
from promo_finder.pairwise_comparison.schemas import PairChoice, PairOutcome, PairVerdict, PairwiseResult

logger = logging.getLogger(__name__)

_PAIR_MAX_OUTPUT_TOKENS = 4000


def compare_in_pairs(pool: list[JudgedCandidate], context: JudgeContext) -> PairwiseResult:
    """`pool` is ordered best first by final score; each promo meets its nearest neighbours in that order."""
    promo_ids = [judged.candidate.candidate_id for judged in pool]
    if len(pool) < 2:
        return PairwiseResult(outcomes=[], strength_by_promo_id={promo_id: 0.5 for promo_id in promo_ids}, failed_pairs=0)
    clips = {judged.candidate.candidate_id: render_clip(judged.candidate, context.show, context.bible, is_anonymized=True) for judged in pool}
    pairs = select_pairs(len(pool), config.PAIRWISE_NEIGHBOUR_WINDOW)
    with ThreadPoolExecutor(max_workers=config.LLM_MAX_PARALLEL_CALLS) as executor:
        results = list(executor.map(lambda pair: _judge_pair(pool[pair[0]], pool[pair[1]], clips, context), pairs))
    outcomes = [outcome for outcome in results if outcome is not None]
    if len(outcomes) < len(pairs):
        logger.warning("Pairwise: %d of %d pairs failed and were skipped", len(pairs) - len(outcomes), len(pairs))
    strengths = fit_bradley_terry(promo_ids, outcomes)
    return PairwiseResult(outcomes=outcomes, strength_by_promo_id=_scale_to_unit_range(strengths), failed_pairs=len(pairs) - len(outcomes))


def select_pairs(pool_size: int, neighbour_window: int) -> list[tuple[int, int]]:
    return [(first, second) for first in range(pool_size) for second in range(first + 1, min(pool_size, first + neighbour_window + 1))]


def fit_bradley_terry(promo_ids: list[str], outcomes: list[PairOutcome]) -> dict[str, float]:
    """Bradley-Terry strengths by the MM algorithm. Each promo also gets one virtual tie against an average
    opponent (strength 1), so a promo that won or lost everything still gets a finite strength."""
    wins = {promo_id: 0.5 for promo_id in promo_ids}
    for outcome in outcomes:
        wins[outcome.first_promo_id] += outcome.first_score
        wins[outcome.second_promo_id] += 1.0 - outcome.first_score
    strengths = {promo_id: 1.0 for promo_id in promo_ids}
    for _ in range(config.BRADLEY_TERRY_ITERATIONS):
        updated = {}
        for promo_id in promo_ids:
            denominator = 1.0 / (strengths[promo_id] + 1.0)
            for outcome in outcomes:
                if promo_id in (outcome.first_promo_id, outcome.second_promo_id):
                    opponent_id = outcome.second_promo_id if promo_id == outcome.first_promo_id else outcome.first_promo_id
                    denominator += 1.0 / (strengths[promo_id] + strengths[opponent_id])
            updated[promo_id] = wins[promo_id] / denominator
        geometric_mean = math.exp(sum(math.log(value) for value in updated.values()) / len(updated))
        strengths = {promo_id: value / geometric_mean for promo_id, value in updated.items()}
    return strengths


def _judge_pair(first: JudgedCandidate, second: JudgedCandidate, clips: dict[str, str], context: JudgeContext) -> PairOutcome | None:
    first_id, second_id = first.candidate.candidate_id, second.candidate.candidate_id
    first_shown_first = _ask(clips[first_id], first.candidate.duration_s, clips[second_id], second.candidate.duration_s, context)
    second_shown_first = _ask(clips[second_id], second.candidate.duration_s, clips[first_id], first.candidate.duration_s, context)
    if first_shown_first is None or second_shown_first is None:
        return None
    # Position bias: a win counts only when it survives swapping the order; a split is a tie.
    first_wins_count = int(first_shown_first.winner == PairChoice.A) + int(second_shown_first.winner == PairChoice.B)
    return PairOutcome(
        first_promo_id=first_id,
        second_promo_id=second_id,
        first_score={2: 1.0, 1: 0.5, 0: 0.0}[first_wins_count],
        reason_first_shown_first=first_shown_first.reason,
        reason_second_shown_first=second_shown_first.reason,
    )


def _ask(clip_a: str, duration_a_s: float, clip_b: str, duration_b_s: float, context: JudgeContext) -> PairVerdict | None:
    return context.llm_service.call_json(
        prompt_name=prompts.PROMPT_NAME, prompt_version=prompts.PROMPT_VERSION, model=context.model_settings.judge_model,
        effort="low", system_prompt=prompts.SYSTEM_PROMPT,
        user_prompt=prompts.build_user_prompt(clip_a, duration_a_s, clip_b, duration_b_s),
        output_model=PairVerdict, max_output_tokens=_PAIR_MAX_OUTPUT_TOKENS,
    )


def _scale_to_unit_range(strengths: dict[str, float]) -> dict[str, float]:
    log_strengths = {promo_id: math.log(value) for promo_id, value in strengths.items()}
    lowest, highest = min(log_strengths.values()), max(log_strengths.values())
    if highest - lowest < 1e-9:
        return {promo_id: 0.5 for promo_id in strengths}
    return {promo_id: round((value - lowest) / (highest - lowest), 4) for promo_id, value in log_strengths.items()}
