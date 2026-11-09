"""Final score, a diverse posting order, and the why-it-works line."""

import logging

from promo_finder import config
from promo_finder.candidates.schemas import Candidate
from promo_finder.cold_viewer.schemas import JudgedCandidate
from promo_finder.config import ModelSettings
from promo_finder.llm.service import LlmService
from promo_finder.ranking import prompts
from promo_finder.ranking.schemas import PromoMoment, PromoSegmentOutput, RankedPromo, WhyLineReply
from promo_finder.screenplay.schemas import Block, Show

logger = logging.getLogger(__name__)


def score_final(judged: JudgedCandidate, is_cold_viewer_enabled: bool) -> float:
    candidate = judged.candidate
    if not is_cold_viewer_enabled or judged.viewer_reaction is None or judged.grader_verdict is None or judged.opening_reaction is None:
        return candidate.prefilter_score
    reaction, verdict = judged.viewer_reaction, judged.grader_verdict
    appeal = 0.5 * (reaction.want_next_score / 5) + 0.5 * (judged.opening_reaction.stop_strength / 5)
    final_score = (
        config.FINAL_APPEAL_WEIGHT * appeal
        + config.FINAL_COMPREHENSION_WEIGHT * (verdict.comprehension / 2)
        + config.FINAL_CLIFFHANGER_MATCH_WEIGHT * (verdict.cliffhanger_match / 2)
        + config.FINAL_HOOK_WEIGHT * candidate.hook_score
        + config.FINAL_SHAPE_WEIGHT * candidate.shape_score
    )
    early_episode_factor = config.EARLY_EPISODE_FINAL_BASE + (1 - config.EARLY_EPISODE_FINAL_BASE) * candidate.early_episode_score
    return round(final_score * early_episode_factor, 4)


def order_for_posting(
    judged_candidates: list[JudgedCandidate], is_cold_viewer_enabled: bool, pairwise_strength_by_promo_id: dict[str, float] | None = None
) -> list[tuple[JudgedCandidate, float]]:
    """Greedy diversity order: each pick is the best mix of its relevance and difference from earlier picks.
    Relevance blends the head-to-head strength (when the promo was in the pairwise pool) with the final score."""
    strengths = pairwise_strength_by_promo_id or {}
    remaining = [(judged, score_final(judged, is_cold_viewer_enabled)) for judged in judged_candidates]
    relevance = {judged.candidate.candidate_id: _blend_relevance(final_score, strengths.get(judged.candidate.candidate_id)) for judged, final_score in remaining}
    ordered: list[tuple[JudgedCandidate, float]] = []
    while remaining:
        best_index = max(
            range(len(remaining)),
            key=lambda index: config.DIVERSITY_RELEVANCE_WEIGHT * relevance[remaining[index][0].candidate.candidate_id]
            - config.DIVERSITY_SIMILARITY_WEIGHT * _max_similarity(remaining[index][0], [judged for judged, _ in ordered]),
        )
        ordered.append(remaining.pop(best_index))
    return ordered


def _blend_relevance(final_score: float, pairwise_strength: float | None) -> float:
    if pairwise_strength is None:
        return (1 - config.PAIRWISE_BLEND_WEIGHT) * final_score
    return config.PAIRWISE_BLEND_WEIGHT * pairwise_strength + (1 - config.PAIRWISE_BLEND_WEIGHT) * final_score


def build_ranked_promos(
    ordered: list[tuple[JudgedCandidate, float]],
    show: Show,
    llm_service: LlmService,
    model_settings: ModelSettings,
    pairwise_strength_by_promo_id: dict[str, float] | None = None,
) -> list[RankedPromo]:
    blocks_by_id = show.build_block_index()
    why_lines = _write_why_lines([judged for judged, _ in ordered[: config.WHY_LINE_LIMIT]], blocks_by_id, llm_service, model_settings)
    strengths = pairwise_strength_by_promo_id or {}
    return [
        _to_ranked_promo(rank, judged, final_score, blocks_by_id, why_lines.get(judged.candidate.candidate_id, ""), strengths.get(judged.candidate.candidate_id))
        for rank, (judged, final_score) in enumerate(ordered, start=1)
    ]


def format_timestamp(seconds: float) -> str:
    minutes, remainder_s = divmod(seconds, 60)
    return f"{int(minutes):02d}:{remainder_s:05.2f}"


def _max_similarity(judged: JudgedCandidate, picked: list[JudgedCandidate]) -> float:
    if not picked:
        return 0.0
    return max(_similarity(judged, other) for other in picked)


def _similarity(first: JudgedCandidate, second: JudgedCandidate) -> float:
    first_candidate, second_candidate = first.candidate, second.candidate
    is_same_arc = first_candidate.arc_id is not None and first_candidate.arc_id == second_candidate.arc_id
    is_same_hook_type = first_candidate.hook_type is not None and first_candidate.hook_type == second_candidate.hook_type
    is_same_episode = first_candidate.segments[0].episode_number == second_candidate.segments[0].episode_number
    return 0.4 * is_same_arc + 0.3 * is_same_hook_type + 0.3 * is_same_episode


def _write_why_lines(
    judged_candidates: list[JudgedCandidate], blocks_by_id: dict[str, Block], llm_service: LlmService, model_settings: ModelSettings
) -> dict[str, str]:
    if not judged_candidates:
        return {}
    briefs = [_build_brief(judged, blocks_by_id) for judged in judged_candidates]
    reply = llm_service.call_json(
        prompt_name=prompts.PROMPT_NAME, prompt_version=prompts.PROMPT_VERSION, model=model_settings.viewer_model, effort="low",
        system_prompt=prompts.SYSTEM_PROMPT, user_prompt=prompts.build_user_prompt(briefs), output_model=WhyLineReply,
    )
    if reply is None:
        logger.error("Why-it-works call failed; promos keep a templated line")
        return {}
    known_ids = {judged.candidate.candidate_id for judged in judged_candidates}
    return {line.promo_id: line.why_it_works for line in reply.lines if line.promo_id in known_ids}


def _build_brief(judged: JudgedCandidate, blocks_by_id: dict[str, Block]) -> str:
    candidate = judged.candidate
    reaction, opening = judged.viewer_reaction, judged.opening_reaction
    reaction_text = (
        f"first 3 s stop strength: {opening.stop_strength}/5 ({opening.reason}); left asking: {reaction.open_question}; "
        f"wants next: {reaction.want_next_score}/5"
        if reaction and opening else "no viewer test"
    )
    return (
        f"promo_id: {candidate.candidate_id}\n"
        f"hook ({candidate.hook_type}): {build_hook_text(candidate, blocks_by_id)}\n"
        f"cliffhanger: {blocks_by_id[candidate.cliffhanger_block_id].text}\n"
        f"intended question: {candidate.open_question}\n"
        f"viewer: {reaction_text}"
    )


def _to_ranked_promo(
    rank: int, judged: JudgedCandidate, final_score: float, blocks_by_id: dict[str, Block], why_line: str, pairwise_strength: float | None
) -> RankedPromo:
    candidate = judged.candidate
    hook_start_block = blocks_by_id[candidate.hook_start_block_id]
    cliffhanger_block = blocks_by_id[candidate.cliffhanger_block_id]
    verdict = judged.grader_verdict
    reaction = judged.viewer_reaction
    scores = {
        "hook": candidate.hook_score,
        "cliffhanger": candidate.cliffhanger_score,
        "shape": candidate.shape_score,
        "early_episode": candidate.early_episode_score,
        "prefilter": candidate.prefilter_score,
    }
    if pairwise_strength is not None:
        scores["pairwise"] = pairwise_strength
    if verdict and reaction and judged.opening_reaction:
        scores.update({
            "comprehension": verdict.comprehension,
            "cliffhanger_match": verdict.cliffhanger_match,
            "want_next": reaction.want_next_score,
            "opening_stop_strength": judged.opening_reaction.stop_strength,
        })
    return RankedPromo(
        rank=rank,
        promo_id=candidate.candidate_id,
        kind=candidate.kind.value,
        segments=[
            PromoSegmentOutput(
                episode_number=segment.episode_number, start=format_timestamp(segment.start_s), end=format_timestamp(segment.end_s),
                start_s=round(segment.start_s, 2), end_s=round(segment.end_s, 2),
            )
            for segment in candidate.segments
        ],
        duration_s=candidate.duration_s,
        hook=_to_moment(hook_start_block, build_hook_text(candidate, blocks_by_id)),
        cliffhanger=_to_moment(cliffhanger_block, cliffhanger_block.text),
        open_question=candidate.open_question,
        why_it_works=why_line or f"Opens on a {candidate.hook_type or 'strong'} moment and cuts on: {candidate.open_question}",
        final_score=final_score,
        scores=scores,
        hook_evidence=candidate.hook_evidence.value if candidate.hook_evidence else None,
        hook_type=candidate.hook_type,
        arc_id=candidate.arc_id,
        flags=candidate.flags,
        viewer_said=reaction.what_is_happening if reaction else "",
    )


def _to_moment(block: Block, text: str) -> PromoMoment:
    return PromoMoment(block_id=block.block_id, episode_number=block.episode_number, at=format_timestamp(block.start_s), text=text)


def build_hook_text(candidate: Candidate, blocks_by_id: dict[str, Block]) -> str:
    """The whole hook sentence: the tagged line plus the lines of the same speaker it continues."""
    start_block = blocks_by_id[candidate.hook_start_block_id]
    hook_block = blocks_by_id[candidate.hook_block_id]
    if start_block.block_id == hook_block.block_id:
        return hook_block.text
    sentence_blocks = sorted(
        (
            block for block in blocks_by_id.values()
            if block.episode_number == hook_block.episode_number and block.speaker == hook_block.speaker
            and start_block.start_s <= block.start_s <= hook_block.start_s
        ),
        key=lambda block: block.start_s,
    )
    return " ".join(block.text for block in sentence_blocks) or hook_block.text
