"""The hard rules every promo must pass, checked against the parsed screenplay (never against LLM output)."""

from promo_finder import config
from promo_finder.candidates.schemas import RejectionReason, Segment
from promo_finder.screenplay.schemas import Block, BlockKind

_TIME_TOLERANCE_S = 0.05


def check_promo_rules(
    segments: list[Segment],
    hook_block: Block,
    cliffhanger_block: Block,
    answer_block: Block | None,
    blocks_by_episode: dict[int, list[Block]],
    recap_block_ids: set[str],
) -> list[RejectionReason]:
    reasons: list[RejectionReason] = []
    if not segments:
        return [RejectionReason.COMPRESSION_FAILED]
    duration_s = sum(segment.duration_s for segment in segments)
    if duration_s < config.PROMO_MIN_DURATION_S - _TIME_TOLERANCE_S:
        reasons.append(RejectionReason.TOO_SHORT)
    if duration_s > config.PROMO_MAX_DURATION_S + _TIME_TOLERANCE_S:
        reasons.append(RejectionReason.TOO_LONG)
    if len(segments) > config.MAX_SEGMENTS:
        reasons.append(RejectionReason.TOO_MANY_SEGMENTS)
    if any(segment.duration_s < config.MIN_SEGMENT_DURATION_S - _TIME_TOLERANCE_S for segment in segments):
        reasons.append(RejectionReason.SEGMENT_TOO_SHORT)
    if not _is_hook_in_window(segments[0], hook_block):
        reasons.append(RejectionReason.HOOK_LATE)
    last_segment = segments[-1]
    if _is_answer_inside(last_segment, cliffhanger_block, answer_block):
        reasons.append(RejectionReason.ANSWER_INSIDE)
    if _has_dialogue_after_cliffhanger(last_segment, cliffhanger_block, blocks_by_episode[last_segment.episode_number]):
        reasons.append(RejectionReason.DIALOGUE_AFTER_CLIFFHANGER)
    if any(_cuts_through_dialogue(segment, blocks_by_episode[segment.episode_number]) for segment in segments):
        reasons.append(RejectionReason.CUT_THROUGH_LINE)
    if any(_contains_recap(segment, blocks_by_episode[segment.episode_number], recap_block_ids) for segment in segments):
        reasons.append(RejectionReason.RECAP_INSIDE)
    return reasons


def _is_hook_in_window(first_segment: Segment, hook_block: Block) -> bool:
    if hook_block.episode_number != first_segment.episode_number:
        return False
    hook_offset_s = hook_block.start_s - first_segment.start_s
    return -_TIME_TOLERANCE_S <= hook_offset_s <= config.HOOK_WINDOW_S + _TIME_TOLERANCE_S


def _is_answer_inside(last_segment: Segment, cliffhanger_block: Block, answer_block: Block | None) -> bool:
    if answer_block is None or answer_block.episode_number != last_segment.episode_number:
        return False
    is_after_cliffhanger = answer_block.start_s > cliffhanger_block.start_s + _TIME_TOLERANCE_S
    return is_after_cliffhanger and last_segment.start_s <= answer_block.start_s < last_segment.end_s - _TIME_TOLERANCE_S


def _has_dialogue_after_cliffhanger(last_segment: Segment, cliffhanger_block: Block, episode_blocks: list[Block]) -> bool:
    """Nothing may be said after the cliffhanger line: a reply can resolve it."""
    for block in episode_blocks:
        if block.kind != BlockKind.DIALOGUE or block.block_id == cliffhanger_block.block_id:
            continue
        starts_after_cliffhanger = block.start_s > cliffhanger_block.start_s + _TIME_TOLERANCE_S
        if starts_after_cliffhanger and block.start_s < last_segment.end_s - _TIME_TOLERANCE_S:
            return True
    return False


def _cuts_through_dialogue(segment: Segment, episode_blocks: list[Block]) -> bool:
    for block in episode_blocks:
        if block.kind != BlockKind.DIALOGUE:
            continue
        is_cut_at_start = block.start_s < segment.start_s - _TIME_TOLERANCE_S < block.end_s - 0.1
        is_cut_at_end = block.start_s < segment.end_s - 0.1 and block.end_s > segment.end_s + _TIME_TOLERANCE_S
        if is_cut_at_start or is_cut_at_end:
            return True
    return False


def _contains_recap(segment: Segment, episode_blocks: list[Block], recap_block_ids: set[str]) -> bool:
    return any(
        block.block_id in recap_block_ids and segment.start_s <= block.start_s < segment.end_s - _TIME_TOLERANCE_S
        for block in episode_blocks
    )
