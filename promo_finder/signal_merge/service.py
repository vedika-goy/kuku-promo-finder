"""Merge the LLM tags with the model-free emotion and sound signals into one scored map of lines."""

import logging

from promo_finder import config
from promo_finder.config import ModelSettings, RunOptions
from promo_finder.line_features.schemas import LineFeatures, SoundType
from promo_finder.llm.service import LlmService
from promo_finder.screenplay.schemas import Block, BlockKind, Episode, Show
from promo_finder.signal_merge import prompts
from promo_finder.signal_merge.schemas import DisputedReviewReply, DisputedVerdict, HookEvidence, ScoredLine
from promo_finder.story_bible.schemas import SpoilerLevel, StoryBible
from promo_finder.story_bible.service import render_bible_for_prompt
from promo_finder.tagging.schemas import ANSWERED_LATER, CliffhangerTag, CliffhangerType, EpisodeTags, HookTag, HookType

logger = logging.getLogger(__name__)

_SOUND_HOOK_VALUE = {SoundType.IMPACT: 1.0, SoundType.STING: 1.0, SoundType.GASP: 0.75, SoundType.ENTRANCE: 0.5}
_DISPUTED_SOUND_TYPES = {SoundType.IMPACT, SoundType.STING}
_DISPUTED_NEIGHBOUR_S = 1.5
_REVIEW_CONTEXT_S = 8.0


def merge_signals(
    show: Show,
    features_by_block_id: dict[str, LineFeatures],
    tags_by_episode: dict[int, EpisodeTags],
    bible: StoryBible,
    llm_service: LlmService,
    model_settings: ModelSettings,
    run_options: RunOptions,
) -> dict[str, ScoredLine]:
    reviewed_hooks: dict[str, HookTag] = {}
    reviewed_cliffhangers: dict[str, CliffhangerTag] = {}
    if run_options.is_emotion_signal_enabled:
        disputed_block_ids = find_disputed_block_ids(show, features_by_block_id, tags_by_episode)
        if disputed_block_ids:
            reviewed_hooks, reviewed_cliffhangers = review_disputed_lines(disputed_block_ids, show, bible, llm_service, model_settings)
    line_map: dict[str, ScoredLine] = {}
    for episode in show.episodes:
        tags = tags_by_episode[episode.episode_number]
        line_map.update(
            _score_episode(episode, features_by_block_id, tags, reviewed_hooks, reviewed_cliffhangers, run_options.is_emotion_signal_enabled)
        )
    return line_map


def find_disputed_block_ids(
    show: Show, features_by_block_id: dict[str, LineFeatures], tags_by_episode: dict[int, EpisodeTags]
) -> list[str]:
    """Lines the signal calls intense that have no LLM tag nearby, strongest first."""
    disputed: list[tuple[float, str]] = []
    for episode in show.episodes:
        tags = tags_by_episode[episode.episode_number]
        if tags.is_untagged:
            continue
        blocks_by_id = {block.block_id: block for block in episode.blocks}
        tagged_starts = [blocks_by_id[tag.block_id].start_s for tag in [*tags.hooks, *tags.cliffhangers] if tag.block_id in blocks_by_id]
        for block in episode.content_blocks:
            feature = features_by_block_id[block.block_id]
            is_signal_strong = (
                feature.z_in_episode >= config.DISPUTED_Z_MIN or feature.sound_type in _DISPUTED_SOUND_TYPES or feature.is_cliffhanger_cue
            )
            is_near_tag = any(abs(block.start_s - tagged_start_s) <= _DISPUTED_NEIGHBOUR_S for tagged_start_s in tagged_starts)
            if is_signal_strong and not is_near_tag and not feature.recap_of_block_id:
                disputed.append((feature.z_in_episode, block.block_id))
    disputed.sort(reverse=True)
    return [block_id for _, block_id in disputed[: config.DISPUTED_REVIEW_LIMIT]]


def review_disputed_lines(
    disputed_block_ids: list[str], show: Show, bible: StoryBible, llm_service: LlmService, model_settings: ModelSettings
) -> tuple[dict[str, HookTag], dict[str, CliffhangerTag]]:
    block_index = show.build_block_index()
    snippets = [(block_id, _render_context_snippet(block_index[block_id], show)) for block_id in disputed_block_ids]
    reply = llm_service.call_json(
        prompt_name=prompts.PROMPT_NAME,
        prompt_version=prompts.PROMPT_VERSION,
        model=model_settings.tagging_model,
        effort="medium",
        system_prompt=prompts.SYSTEM_PROMPT,
        user_prompt=prompts.build_user_prompt(render_bible_for_prompt(bible), snippets),
        output_model=DisputedReviewReply,
    )
    if reply is None:
        logger.error("Disputed-line review failed; those lines stay untagged")
        return {}, {}
    hooks: dict[str, HookTag] = {}
    cliffhangers: dict[str, CliffhangerTag] = {}
    for review in reply.reviews:
        if review.block_id not in disputed_block_ids:
            logger.info("Review returned unknown block id %s; ignored", review.block_id)
            continue
        if review.verdict == DisputedVerdict.HOOK:
            hooks[review.block_id] = HookTag(
                block_id=review.block_id, strength=review.strength, hook_type=review.hook_type or HookType.SHOCK_LINE,
                trigger_block_id=None, needs_context=[], spoiler_level=SpoilerLevel.NONE, reason="confirmed in the disputed-line review",
            )
        elif review.verdict == DisputedVerdict.CLIFFHANGER:
            cliffhangers[review.block_id] = CliffhangerTag(
                block_id=review.block_id, strength=review.strength,
                cliffhanger_type=review.cliffhanger_type or CliffhangerType.CUT_BEFORE_ANSWER,
                open_question=review.open_question, answered_at=ANSWERED_LATER,
            )
    return hooks, cliffhangers


def _score_episode(
    episode: Episode,
    features_by_block_id: dict[str, LineFeatures],
    tags: EpisodeTags,
    reviewed_hooks: dict[str, HookTag],
    reviewed_cliffhangers: dict[str, CliffhangerTag],
    is_emotion_signal_enabled: bool,
) -> dict[str, ScoredLine]:
    hooks_by_id = {tag.block_id: tag for tag in tags.hooks}
    cliffhangers_by_id = {tag.block_id: tag for tag in tags.cliffhangers}
    arc_lines_by_id = {arc_line.block_id: arc_line for arc_line in tags.arc_lines}
    scored_lines = {}
    for block in episode.blocks:
        feature = features_by_block_id.get(block.block_id)
        hook_signal, cliffhanger_signal = _measure_signals(block, episode, features_by_block_id, is_emotion_signal_enabled)
        hook_tag = hooks_by_id.get(block.block_id) or reviewed_hooks.get(block.block_id)
        cliffhanger_tag = cliffhangers_by_id.get(block.block_id) or reviewed_cliffhangers.get(block.block_id)
        boundary = _measure_boundary(feature, block, episode)
        arc_line = arc_lines_by_id.get(block.block_id)
        scored_lines[block.block_id] = ScoredLine(
            block_id=block.block_id,
            episode_number=episode.episode_number,
            kind=block.kind,
            start_s=block.start_s,
            end_s=block.end_s,
            text=block.text,
            speaker=block.speaker,
            intensity=feature.intensity if feature else None,
            z_in_episode=feature.z_in_episode if feature else None,
            signal_strength=round(hook_signal, 3),
            hook_score=round(_combine_hook_score(hook_tag, hook_signal, tags.is_untagged, is_emotion_signal_enabled), 3),
            cliffhanger_score=round(
                _combine_cliffhanger_score(cliffhanger_tag, boundary, cliffhanger_signal, tags.is_untagged, is_emotion_signal_enabled), 3
            ),
            hook_evidence=_classify_evidence(hook_tag, block.block_id in reviewed_hooks, hook_signal),
            hook_tag=hook_tag,
            cliffhanger_tag=cliffhanger_tag,
            arc_id=arc_line.arc_id if arc_line else None,
            importance=arc_line.importance if arc_line else None,
            is_recap=bool(feature and feature.recap_of_block_id),
        )
    return scored_lines


def _measure_signals(
    block: Block, episode: Episode, features_by_block_id: dict[str, LineFeatures], is_emotion_signal_enabled: bool
) -> tuple[float, float]:
    """(hook signal, cliffhanger signal) from the content blocks overlapping a window around the line."""
    if not is_emotion_signal_enabled:
        return 0.0, 0.0
    window_start_s = block.start_s - config.SIGNAL_WINDOW_BEFORE_S
    window_end_s = block.end_s + config.SIGNAL_WINDOW_AFTER_S
    hook_signal = 0.0
    cliffhanger_signal = 0.0
    for neighbour in episode.content_blocks:
        if neighbour.end_s < window_start_s or neighbour.start_s > window_end_s:
            continue
        feature = features_by_block_id[neighbour.block_id]
        emotion_spike = min(1.0, max(0.0, (feature.z_in_episode - 1.0) / 2.0))
        sound_value = _SOUND_HOOK_VALUE.get(feature.sound_type, 0.0)
        hook_signal = max(hook_signal, emotion_spike, sound_value)
        is_cliffhanger_sound = feature.is_cliffhanger_cue or feature.sound_type == SoundType.STING
        tension_value = 1.0 if (feature.has_tension_tag or is_cliffhanger_sound) else 0.5 * feature.intensity
        cliffhanger_signal = max(cliffhanger_signal, tension_value)
    return hook_signal, cliffhanger_signal


def _measure_boundary(feature: LineFeatures | None, block: Block, episode: Episode) -> float:
    if feature is not None:
        if feature.is_episode_tail:
            return 1.0
        return 0.5 if feature.is_scene_end else 0.0
    is_tail = block.end_s >= episode.duration_s - config.EPISODE_TAIL_S
    return 1.0 if is_tail else 0.0


def _combine_hook_score(hook_tag: HookTag | None, signal: float, is_untagged: bool, is_emotion_signal_enabled: bool) -> float:
    llm_value = hook_tag.strength / 5 if hook_tag else 0.0
    if is_untagged:
        return signal
    if not is_emotion_signal_enabled:
        return llm_value
    return config.HOOK_LLM_WEIGHT * llm_value + config.HOOK_SIGNAL_WEIGHT * signal


def _combine_cliffhanger_score(
    cliffhanger_tag: CliffhangerTag | None, boundary: float, signal: float, is_untagged: bool, is_emotion_signal_enabled: bool
) -> float:
    llm_value = cliffhanger_tag.strength / 5 if cliffhanger_tag else 0.0
    if is_untagged:
        return 0.5 * boundary + 0.5 * signal
    if not is_emotion_signal_enabled:
        return 0.75 * llm_value + 0.25 * boundary
    return (
        config.CLIFFHANGER_LLM_WEIGHT * llm_value
        + config.CLIFFHANGER_BOUNDARY_WEIGHT * boundary
        + config.CLIFFHANGER_SIGNAL_WEIGHT * signal
    )


def _classify_evidence(hook_tag: HookTag | None, is_reviewed: bool, signal: float) -> HookEvidence | None:
    if hook_tag is None:
        return None
    if is_reviewed:
        return HookEvidence.SIGNAL_ONLY
    return HookEvidence.CORROBORATED if signal >= config.CORROBORATION_SIGNAL_MIN else HookEvidence.CONTEXT_HOOK


def _render_context_snippet(target: Block, show: Show) -> str:
    episode = show.find_episode(target.episode_number)
    lines = []
    for block in episode.blocks:
        if abs(block.start_s - target.start_s) > _REVIEW_CONTEXT_S:
            continue
        marker = ">>" if block.block_id == target.block_id else "  "
        kind_label = "SHOT" if block.kind == BlockKind.CAMERA else block.kind.value.upper()
        speaker = f"{block.speaker} ({', '.join(block.emotion_tags)}): " if block.kind == BlockKind.DIALOGUE else ""
        lines.append(f"{marker} {block.block_id} | {kind_label} | {speaker}{block.text}")
    return f"Episode {episode.episode_number}\n" + "\n".join(lines)
