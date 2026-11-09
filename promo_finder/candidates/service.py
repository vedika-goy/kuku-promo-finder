"""Turn scored lines into promo candidates: hook x cliffhanger windows snapped to shots, stitched, compressed."""

import logging
import re
import statistics
from dataclasses import dataclass, field

from promo_finder import config
from promo_finder.candidates.schemas import Candidate, CandidateKind, CandidateSet, RejectionReason, Segment
from promo_finder.line_features.schemas import LineFeatures
from promo_finder.line_features.service import compute_recap_end_s
from promo_finder.promo_rules.service import check_promo_rules
from promo_finder.screenplay.schemas import Block, BlockKind, Episode, Show
from promo_finder.signal_merge.schemas import ScoredLine
from promo_finder.story_bible.schemas import SpoilerLevel
from promo_finder.tagging.schemas import LineImportance

logger = logging.getLogger(__name__)

_CONTINUATION_OPENING = re.compile(r"^\s*(\.\.\.|…)")
_BLACK_SCREEN = re.compile(r"\b(cut|cuts|fade|fades)\s+to\s+black\b", re.I)
_IMPORTANCE_RANK = {LineImportance.ESSENTIAL: 3, LineImportance.SUPPORTING: 2, None: 1}
_CONTIGUITY_TOLERANCE_S = 0.05


@dataclass
class _Unit:
    """A stretch between two cut points that never splits a line of dialogue."""

    episode_number: int
    start_s: float
    end_s: float
    content_block_ids: list[str] = field(default_factory=list)
    importance: int = 1
    intensity: float = 0.0
    is_locked: bool = False

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


@dataclass
class _ShowContext:
    show: Show
    line_map: dict[str, ScoredLine]
    blocks_by_id: dict[str, Block]
    blocks_by_episode: dict[int, list[Block]]
    recap_block_ids: set[str]
    recap_end_by_episode: dict[int, float]


def build_candidates(show: Show, line_map: dict[str, ScoredLine], features_by_block_id: dict[str, LineFeatures]) -> CandidateSet:
    context = _build_context(show, line_map, features_by_block_id)
    hooks = [line for line in line_map.values() if _is_eligible_hook(line)]
    cliffhangers = [line for line in line_map.values() if _is_eligible_cliffhanger(line)]
    candidates: list[Candidate] = []
    for hook_line in hooks:
        for cliffhanger_line in cliffhangers:
            candidate = _build_pair_candidate(hook_line, cliffhanger_line, context)
            if candidate is not None:
                candidates.append(candidate)
    return _select_candidates(candidates)


def _is_eligible_hook(line: ScoredLine) -> bool:
    if line.is_recap:
        return False
    # Why not the score alone: a quiet reveal rated 4 by the LLM has no emotion spike and scores 0.48;
    # it goes to the cold viewer instead, which can reject it.
    is_strong_llm_tag = line.hook_tag is not None and line.hook_tag.strength >= config.LLM_STRENGTH_ALWAYS_ELIGIBLE
    return line.hook_score >= config.HOOK_SCORE_MIN or is_strong_llm_tag


def _is_eligible_cliffhanger(line: ScoredLine) -> bool:
    if line.is_recap:
        return False
    is_strong_llm_tag = line.cliffhanger_tag is not None and line.cliffhanger_tag.strength >= config.LLM_STRENGTH_ALWAYS_ELIGIBLE
    return line.cliffhanger_score >= config.CLIFFHANGER_SCORE_MIN or is_strong_llm_tag


def _is_held_spoiler(candidate: Candidate) -> bool:
    """Major spoilers are held back, except in the free episodes: a new viewer sees those within minutes anyway."""
    return candidate.spoiler_level == SpoilerLevel.MAJOR and candidate.segments[0].episode_number > config.FREE_EPISODE_COUNT


def restore_dropped_line(
    candidate: Candidate, show: Show, line_map: dict[str, ScoredLine], features_by_block_id: dict[str, LineFeatures]
) -> Candidate | None:
    """Add back the most important line that compression dropped, if the promo still passes every rule."""
    if not candidate.dropped_block_ids:
        return None
    context = _build_context(show, line_map, features_by_block_id)
    restored_block = context.blocks_by_id[
        max(candidate.dropped_block_ids, key=lambda block_id: (_IMPORTANCE_RANK[line_map[block_id].importance], line_map[block_id].intensity or 0.0))
    ]
    episode = show.find_episode(restored_block.episode_number)
    episode_segments = [segment for segment in candidate.segments if segment.episode_number == episode.episode_number]
    gap_start_s = max((segment.end_s for segment in episode_segments if segment.end_s <= restored_block.start_s), default=None)
    gap_end_s = min((segment.start_s for segment in episode_segments if segment.start_s > restored_block.start_s), default=None)
    if gap_start_s is None or gap_end_s is None:
        return None
    gap_units = _build_units(episode, gap_start_s, gap_end_s, line_map, set())
    restored_unit = next((unit for unit in gap_units if unit.start_s <= restored_block.start_s < unit.end_s), None)
    if restored_unit is None:
        return None
    all_units = [_segment_to_unit(segment, context) for segment in candidate.segments] + [restored_unit]
    all_units.sort(key=lambda unit: (unit.episode_number, unit.start_s))
    segments = _units_to_segments(all_units)
    cliffhanger_line = line_map[candidate.cliffhanger_block_id]
    reasons = check_promo_rules(
        segments, context.blocks_by_id[candidate.hook_start_block_id], context.blocks_by_id[candidate.cliffhanger_block_id],
        _find_answer_block(cliffhanger_line, context), context.blocks_by_episode, context.recap_block_ids,
    )
    if reasons:
        return None
    restored_ids = set(restored_unit.content_block_ids)
    repaired = candidate.model_copy(deep=True, update={
        "candidate_id": f"{candidate.candidate_id}-R",
        "segments": segments,
        "content_block_ids": [block_id for unit in all_units for block_id in unit.content_block_ids],
        "dropped_block_ids": [block_id for block_id in candidate.dropped_block_ids if block_id not in restored_ids],
        "duration_s": round(sum(segment.duration_s for segment in segments), 2),
        "hook_offset_s": round(context.blocks_by_id[candidate.hook_start_block_id].start_s - segments[0].start_s, 2),
    })
    return _score_candidate(repaired, context)


def score_early_episode(episode_number: int) -> float:
    if episode_number <= config.EARLY_EPISODE_FULL_BONUS_THROUGH:
        return 1.0
    if episode_number >= config.EARLY_EPISODE_FLOOR_AT:
        return config.EARLY_EPISODE_FLOOR
    span = config.EARLY_EPISODE_FLOOR_AT - config.EARLY_EPISODE_FULL_BONUS_THROUGH
    progress = (episode_number - config.EARLY_EPISODE_FULL_BONUS_THROUGH) / span
    return round(1.0 - progress * (1.0 - config.EARLY_EPISODE_FLOOR), 3)


def measure_overlap(first: Candidate, second: Candidate) -> float:
    first_ids, second_ids = set(first.content_block_ids), set(second.content_block_ids)
    if not first_ids or not second_ids:
        return 0.0
    return len(first_ids & second_ids) / min(len(first_ids), len(second_ids))


def _build_context(show: Show, line_map: dict[str, ScoredLine], features_by_block_id: dict[str, LineFeatures]) -> _ShowContext:
    return _ShowContext(
        show=show,
        line_map=line_map,
        blocks_by_id=show.build_block_index(),
        blocks_by_episode={episode.episode_number: episode.blocks for episode in show.episodes},
        recap_block_ids={block_id for block_id, feature in features_by_block_id.items() if feature.recap_of_block_id},
        recap_end_by_episode={episode.episode_number: compute_recap_end_s(episode, features_by_block_id) for episode in show.episodes},
    )


def _build_pair_candidate(hook_line: ScoredLine, cliffhanger_line: ScoredLine, context: _ShowContext) -> Candidate | None:
    if cliffhanger_line.episode_number == hook_line.episode_number:
        is_ordered = cliffhanger_line.start_s > hook_line.start_s
        if not is_ordered or cliffhanger_line.end_s - hook_line.start_s > config.MAX_RAW_SPAN_S:
            return None
        return _build_same_episode_candidate(hook_line, cliffhanger_line, context)
    if cliffhanger_line.episode_number == hook_line.episode_number + 1:
        return _build_cross_episode_candidate(hook_line, cliffhanger_line, context)
    return None


def _build_same_episode_candidate(hook_line: ScoredLine, cliffhanger_line: ScoredLine, context: _ShowContext) -> Candidate | None:
    episode = context.show.find_episode(hook_line.episode_number)
    hook_start_block = _find_sentence_start(context.blocks_by_id[hook_line.block_id], episode)
    cliffhanger_block = context.blocks_by_id[cliffhanger_line.block_id]
    answer_block = _find_answer_block(cliffhanger_line, context)
    start_s, start_flags = _snap_start(episode, hook_start_block, _find_trigger_block(hook_line, context))
    end_s, end_flags = _snap_end(episode, cliffhanger_block, answer_block)
    if end_s - start_s < config.PROMO_MIN_DURATION_S - 0.5:
        return None  # too short to be a promo; not worth logging hundreds of these
    locked_ids = {hook_start_block.block_id, hook_line.block_id, cliffhanger_block.block_id}
    units = _build_units(episode, start_s, end_s, context.line_map, locked_ids)
    return _assemble_candidate(units, hook_line, hook_start_block, cliffhanger_line, answer_block, start_flags + end_flags, context)


def _build_cross_episode_candidate(hook_line: ScoredLine, cliffhanger_line: ScoredLine, context: _ShowContext) -> Candidate | None:
    first_episode = context.show.find_episode(hook_line.episode_number)
    second_episode = context.show.find_episode(cliffhanger_line.episode_number)
    recap_end_s = context.recap_end_by_episode.get(second_episode.episode_number, 0.0)
    is_hook_near_end = hook_line.start_s >= first_episode.duration_s - config.CROSS_EPISODE_REACH_S
    is_cliffhanger_near_start = recap_end_s <= cliffhanger_line.start_s <= recap_end_s + config.CROSS_EPISODE_REACH_S
    if not (is_hook_near_end and is_cliffhanger_near_start):
        return None
    hook_start_block = _find_sentence_start(context.blocks_by_id[hook_line.block_id], first_episode)
    cliffhanger_block = context.blocks_by_id[cliffhanger_line.block_id]
    answer_block = _find_answer_block(cliffhanger_line, context)
    start_s, start_flags = _snap_start(first_episode, hook_start_block, _find_trigger_block(hook_line, context))
    second_start_s = _first_shot_start_at_or_after(second_episode, recap_end_s)
    end_s, end_flags = _snap_end(second_episode, cliffhanger_block, answer_block)
    if second_start_s >= end_s:
        return None
    first_units = [
        unit for unit in _build_units(first_episode, start_s, first_episode.duration_s, context.line_map, {hook_start_block.block_id, hook_line.block_id})
        if not _is_black_screen_unit(unit, first_episode)
    ]
    second_units = _build_units(second_episode, second_start_s, end_s, context.line_map, {cliffhanger_block.block_id})
    flags = start_flags + end_flags + (["RECAP_SKIPPED"] if recap_end_s > 0 else [])
    return _assemble_candidate(first_units + second_units, hook_line, hook_start_block, cliffhanger_line, answer_block, flags, context)


def _assemble_candidate(
    units: list[_Unit],
    hook_line: ScoredLine,
    hook_start_block: Block,
    cliffhanger_line: ScoredLine,
    answer_block: Block | None,
    flags: list[str],
    context: _ShowContext,
) -> Candidate | None:
    if not units:
        return None
    compression = _compress_units(units, config.PROMO_MAX_DURATION_S)
    kept_units, dropped_units = compression if compression else (units, [])
    segments = _units_to_segments(kept_units)
    is_cross_episode = hook_line.episode_number != cliffhanger_line.episode_number
    kind = CandidateKind.CROSS_EPISODE if is_cross_episode else (CandidateKind.COMPRESSED if dropped_units else CandidateKind.SINGLE)
    if hook_start_block.block_id != hook_line.block_id:
        flags = [*flags, "HOOK_SENTENCE_JOINED"]
    candidate = Candidate(
        candidate_id=f"P-{hook_line.block_id}-{cliffhanger_line.block_id}",
        kind=kind,
        segments=segments,
        hook_block_id=hook_line.block_id,
        hook_start_block_id=hook_start_block.block_id,
        cliffhanger_block_id=cliffhanger_line.block_id,
        content_block_ids=[block_id for unit in kept_units for block_id in unit.content_block_ids],
        dropped_block_ids=[block_id for unit in dropped_units for block_id in unit.content_block_ids],
        duration_s=round(sum(segment.duration_s for segment in segments), 2),
        hook_offset_s=round(hook_start_block.start_s - segments[0].start_s, 2),
        hook_score=hook_line.hook_score,
        cliffhanger_score=cliffhanger_line.cliffhanger_score,
        hook_evidence=hook_line.hook_evidence,
        spoiler_level=hook_line.hook_tag.spoiler_level if hook_line.hook_tag else SpoilerLevel.NONE,
        needs_context=hook_line.hook_tag.needs_context if hook_line.hook_tag else [],
        open_question=cliffhanger_line.cliffhanger_tag.open_question if cliffhanger_line.cliffhanger_tag else "",
        hook_type=hook_line.hook_tag.hook_type.value if hook_line.hook_tag else None,
        arc_id=hook_line.arc_id or cliffhanger_line.arc_id,
        flags=flags,
    )
    reasons = check_promo_rules(
        segments, hook_start_block, context.blocks_by_id[cliffhanger_line.block_id], answer_block, context.blocks_by_episode, context.recap_block_ids
    )
    if compression is None and candidate.duration_s > config.PROMO_MAX_DURATION_S:
        reasons.append(RejectionReason.COMPRESSION_FAILED)
    candidate.rejection_reasons = reasons
    return _score_candidate(candidate, context) if not reasons else candidate


def _find_sentence_start(hook_block: Block, episode: Episode) -> Block:
    """A line opening with '...' finishes the same speaker's previous line; the hook lands where the sentence starts."""
    sentence_start = hook_block
    dialogue_blocks = [block for block in episode.blocks if block.kind == BlockKind.DIALOGUE]
    while sentence_start.kind == BlockKind.DIALOGUE and _CONTINUATION_OPENING.match(sentence_start.text):
        previous = next((block for block in reversed(dialogue_blocks) if block.start_s < sentence_start.start_s), None)
        is_same_sentence = (
            previous is not None
            and previous.speaker == sentence_start.speaker
            and sentence_start.start_s - previous.end_s <= config.CONTINUATION_GAP_MAX_S
        )
        if not is_same_sentence:
            break
        sentence_start = previous
    return sentence_start


def _snap_start(episode: Episode, hook_block: Block, trigger_block: Block | None) -> tuple[float, list[str]]:
    anchor_s = hook_block.start_s
    trigger_sentence_start = _find_sentence_start(trigger_block, episode) if trigger_block else None
    if trigger_sentence_start and hook_block.start_s - trigger_sentence_start.start_s <= config.HOOK_WINDOW_S:
        anchor_s = trigger_sentence_start.start_s
    shot = _find_shot_at(episode, anchor_s)
    flags: list[str] = []
    if shot and anchor_s - shot.start_s <= config.SHOT_SNAP_TOLERANCE_S and hook_block.start_s - shot.start_s <= config.HOOK_WINDOW_S:
        start_s = shot.start_s
    else:
        start_s = anchor_s
        flags.append("MID_SHOT_START")
    # Never open in the middle of a line: back up to the start of a line still being spoken.
    straddling_starts = [
        block.start_s for block in episode.blocks
        if block.kind == BlockKind.DIALOGUE and block.start_s < start_s - 0.05 and block.end_s > start_s + 0.05
    ]
    if straddling_starts and hook_block.start_s - min(straddling_starts) <= config.HOOK_WINDOW_S:
        start_s = min(straddling_starts)
    # Never open on the tail of a sentence ("...and it isn't you."): start after that fragment instead.
    opening_fragments = [
        block for block in episode.blocks
        if block.kind == BlockKind.DIALOGUE and start_s - 0.05 <= block.start_s < hook_block.start_s
        and _find_sentence_start(block, episode).start_s < start_s - 0.05
    ]
    if opening_fragments:
        start_s = max(block.end_s for block in opening_fragments)
        flags.append("OPENING_FRAGMENT_SKIPPED")
    return start_s, flags


def _snap_end(episode: Episode, cliffhanger_block: Block, answer_block: Block | None) -> tuple[float, list[str]]:
    line_end_s = cliffhanger_block.end_s
    flags: list[str] = []
    shot = _find_shot_at(episode, max(cliffhanger_block.start_s, line_end_s - 0.01))
    if shot and shot.end_s - line_end_s <= config.REACTION_SHOT_MAX_S:
        end_s = max(shot.end_s, line_end_s)
    else:
        end_s = line_end_s + config.MID_SHOT_END_PAD_S
        flags.append("MID_SHOT_END")
    next_shot = next((candidate_shot for candidate_shot in episode.shots if candidate_shot.start_s >= end_s - _CONTIGUITY_TOLERANCE_S), None)
    if next_shot and _is_reaction_shot(next_shot, episode, answer_block):
        end_s = next_shot.end_s
        flags.append("REACTION_SHOT")
    if answer_block and answer_block.episode_number == episode.episode_number and answer_block.start_s > cliffhanger_block.start_s:
        end_s = min(end_s, answer_block.start_s)
    next_dialogue_starts = [
        block.start_s for block in episode.blocks
        if block.kind == BlockKind.DIALOGUE and block.block_id != cliffhanger_block.block_id and block.start_s > cliffhanger_block.start_s + 0.05
    ]
    if next_dialogue_starts:
        end_s = min(end_s, min(next_dialogue_starts))
    return min(max(end_s, line_end_s), episode.duration_s), flags


def _is_reaction_shot(shot: Block, episode: Episode, answer_block: Block | None) -> bool:
    if shot.duration_s > config.REACTION_SHOT_MAX_S or _BLACK_SCREEN.search(shot.text):
        return False
    has_dialogue = any(
        block.kind == BlockKind.DIALOGUE and shot.start_s <= block.start_s < shot.end_s for block in episode.blocks
    )
    is_answer_inside = answer_block is not None and answer_block.episode_number == episode.episode_number and answer_block.start_s < shot.end_s
    return not has_dialogue and not is_answer_inside


def _find_shot_at(episode: Episode, time_s: float) -> Block | None:
    containing_shots = [shot for shot in episode.shots if shot.start_s <= time_s < shot.end_s]
    return max(containing_shots, key=lambda shot: shot.start_s) if containing_shots else None


def _first_shot_start_at_or_after(episode: Episode, time_s: float) -> float:
    if time_s <= 0:
        return 0.0
    return min((shot.start_s for shot in episode.shots if shot.start_s >= time_s - _CONTIGUITY_TOLERANCE_S), default=time_s)


def _find_trigger_block(hook_line: ScoredLine, context: _ShowContext) -> Block | None:
    trigger_id = hook_line.hook_tag.trigger_block_id if hook_line.hook_tag else None
    return context.blocks_by_id.get(trigger_id) if trigger_id else None


def _find_answer_block(cliffhanger_line: ScoredLine, context: _ShowContext) -> Block | None:
    if cliffhanger_line.cliffhanger_tag is None:
        return None
    return context.blocks_by_id.get(cliffhanger_line.cliffhanger_tag.answered_at)


def _build_units(episode: Episode, start_s: float, end_s: float, line_map: dict[str, ScoredLine], locked_block_ids: set[str]) -> list[_Unit]:
    cut_points = sorted({start_s, end_s, *(shot.start_s for shot in episode.shots if start_s < shot.start_s < end_s)})
    dialogue_blocks = [block for block in episode.blocks if block.kind == BlockKind.DIALOGUE]
    # A cut point inside a line of dialogue would split the line, so it is not a cut point.
    cut_points = [
        point for point in cut_points
        if point in (start_s, end_s) or not any(block.start_s < point - 0.05 and block.end_s > point + 0.05 for block in dialogue_blocks)
    ]
    units = [_Unit(episode_number=episode.episode_number, start_s=left, end_s=right) for left, right in zip(cut_points, cut_points[1:])]
    for unit in units:
        _fill_unit(unit, episode, line_map, locked_block_ids)
    if units:
        units[0].is_locked = True
        units[-1].is_locked = True
    return units


def _fill_unit(unit: _Unit, episode: Episode, line_map: dict[str, ScoredLine], locked_block_ids: set[str]) -> None:
    for block in episode.content_blocks:
        if unit.start_s <= block.start_s < unit.end_s:
            unit.content_block_ids.append(block.block_id)
            line = line_map[block.block_id]
            unit.importance = max(unit.importance, _IMPORTANCE_RANK[line.importance])
            unit.intensity = max(unit.intensity, line.intensity or 0.0)
            if block.block_id in locked_block_ids:
                unit.is_locked = True
    if any(block_id in locked_block_ids for block_id in unit.content_block_ids):
        unit.is_locked = True


def _segment_to_unit(segment: Segment, context: _ShowContext) -> _Unit:
    unit = _Unit(episode_number=segment.episode_number, start_s=segment.start_s, end_s=segment.end_s, is_locked=True)
    _fill_unit(unit, context.show.find_episode(segment.episode_number), context.line_map, set())
    return unit


def _is_black_screen_unit(unit: _Unit, episode: Episode) -> bool:
    if unit.content_block_ids:
        return False
    shots = [shot for shot in episode.shots if unit.start_s <= shot.start_s < unit.end_s]
    return bool(shots) and all(_BLACK_SCREEN.search(shot.text) for shot in shots)


def _compress_units(units: list[_Unit], max_duration_s: float) -> tuple[list[_Unit], list[_Unit]] | None:
    """Drop the least important unlocked units until the clip fits. None when it cannot fit."""
    kept = list(units)
    total_s = sum(unit.duration_s for unit in kept)
    dropped: list[_Unit] = []
    droppable = sorted(
        (unit for unit in units if not unit.is_locked and unit.importance < _IMPORTANCE_RANK[LineImportance.ESSENTIAL]),
        key=lambda unit: (unit.importance, unit.intensity, -unit.duration_s),
    )
    is_progressing = True
    while total_s > max_duration_s and is_progressing:
        is_progressing = False
        for unit in droppable:
            if unit in dropped or total_s <= max_duration_s:
                continue
            trial = [kept_unit for kept_unit in kept if kept_unit is not unit]
            if len(_units_to_segments(trial)) > config.MAX_SEGMENTS:
                continue
            kept, total_s = trial, total_s - unit.duration_s
            dropped.append(unit)
            is_progressing = True
    if total_s > max_duration_s:
        return None
    return kept, dropped


def _units_to_segments(units: list[_Unit]) -> list[Segment]:
    segments: list[Segment] = []
    for unit in units:
        last = segments[-1] if segments else None
        is_contiguous = last and last.episode_number == unit.episode_number and abs(last.end_s - unit.start_s) <= _CONTIGUITY_TOLERANCE_S
        if is_contiguous:
            last.end_s = unit.end_s
        else:
            segments.append(Segment(episode_number=unit.episode_number, start_s=unit.start_s, end_s=unit.end_s))
    return segments


def _score_candidate(candidate: Candidate, context: _ShowContext) -> Candidate:
    candidate.shape_score = _score_emotion_shape(candidate, context)
    candidate.duration_fit = round(
        max(0.0, 1.0 - abs(candidate.duration_s - config.PROMO_TARGET_DURATION_S) / (config.PROMO_MAX_DURATION_S - config.PROMO_TARGET_DURATION_S)), 3
    )
    candidate.early_episode_score = score_early_episode(candidate.segments[0].episode_number)
    candidate.prefilter_score = round(
        config.PREFILTER_HOOK_WEIGHT * candidate.hook_score
        + config.PREFILTER_CLIFFHANGER_WEIGHT * candidate.cliffhanger_score
        + config.PREFILTER_SHAPE_WEIGHT * candidate.shape_score
        + config.PREFILTER_DURATION_WEIGHT * candidate.duration_fit
        + config.PREFILTER_EARLY_EPISODE_WEIGHT * candidate.early_episode_score
        - config.PREFILTER_EXTRA_SEGMENT_PENALTY * (len(candidate.segments) - 1)
        - config.PREFILTER_CONTEXT_FACT_PENALTY * len(candidate.needs_context),
        4,
    )
    return candidate


def _score_emotion_shape(candidate: Candidate, context: _ShowContext) -> float:
    """1.0 = opens above the clip's average and ends near its peak; lower when the peak comes early and fades."""
    timeline = [
        (context.blocks_by_id[block_id], context.line_map[block_id].intensity or 0.0) for block_id in candidate.content_block_ids
    ]
    if len(timeline) < 2:
        return 0.5
    intensities = [intensity for _, intensity in timeline]
    mean_intensity = statistics.fmean(intensities)
    peak_intensity = max(intensities)
    opening_peak = max(intensities[: max(1, len(intensities) // 5)])
    closing_peak = max(intensities[-max(1, len(intensities) // 4):])
    opening_score = 1.0 if opening_peak >= mean_intensity else 0.5
    ending_threshold = sorted(intensities)[int(0.8 * (len(intensities) - 1))]
    ending_score = 1.0 if closing_peak >= ending_threshold else 0.4
    peak_index = intensities.index(peak_intensity)
    is_early_peak_fading = peak_index < 0.75 * len(intensities) and closing_peak < peak_intensity - 0.3
    return round((opening_score + ending_score) / 2 * (0.5 if is_early_peak_fading else 1.0), 3)


def _select_candidates(candidates: list[Candidate]) -> CandidateSet:
    passing = [candidate for candidate in candidates if not candidate.rejection_reasons]
    rejected = [candidate for candidate in candidates if candidate.rejection_reasons]
    held = [candidate for candidate in passing if _is_held_spoiler(candidate)]
    eligible = sorted((candidate for candidate in passing if not _is_held_spoiler(candidate)), key=lambda candidate: -candidate.prefilter_score)
    kept: list[Candidate] = []
    for candidate in eligible:
        if any(measure_overlap(candidate, kept_candidate) > config.DEDUP_OVERLAP_MAX for kept_candidate in kept):
            candidate.rejection_reasons = [RejectionReason.DUPLICATE]
            rejected.append(candidate)
        else:
            kept.append(candidate)
    logger.info("Candidates: %d built, %d kept, %d held for spoilers, %d rejected", len(candidates), len(kept), len(held), len(rejected))
    return CandidateSet(kept=kept, held_for_spoilers=sorted(held, key=lambda candidate: -candidate.prefilter_score), rejected=rejected)
