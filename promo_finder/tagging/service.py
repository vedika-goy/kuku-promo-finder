"""Per-episode LLM tagging of hooks, cliffhangers and storyline lines, checked against the real block ids."""

import logging
from concurrent.futures import ThreadPoolExecutor

from promo_finder import config
from promo_finder.config import ModelSettings
from promo_finder.line_features.schemas import LineFeatures
from promo_finder.llm.service import LlmService
from promo_finder.screenplay.schemas import Episode, Show
from promo_finder.story_bible.schemas import StoryBible
from promo_finder.story_bible.service import find_episode_summary, render_bible_for_prompt
from promo_finder.tagging import prompts
from promo_finder.tagging.schemas import ANSWERED_LATER, ANSWERED_NEVER, EpisodeTags

logger = logging.getLogger(__name__)

_TAGGING_MAX_OUTPUT_TOKENS = 16000


def tag_episodes(
    show: Show,
    bible: StoryBible,
    features_by_block_id: dict[str, LineFeatures],
    llm_service: LlmService,
    model_settings: ModelSettings,
) -> dict[int, EpisodeTags]:
    bible_text = render_bible_for_prompt(bible)
    with ThreadPoolExecutor(max_workers=config.LLM_MAX_PARALLEL_CALLS) as executor:
        futures = {
            episode.episode_number: executor.submit(
                tag_episode, episode, show.title, bible_text, bible, features_by_block_id, llm_service, model_settings
            )
            for episode in show.episodes
        }
        return {episode_number: future.result() for episode_number, future in futures.items()}


def tag_episode(
    episode: Episode,
    show_title: str,
    bible_text: str,
    bible: StoryBible,
    features_by_block_id: dict[str, LineFeatures],
    llm_service: LlmService,
    model_settings: ModelSettings,
) -> EpisodeTags:
    recap_block_ids = {
        block.block_id for block in episode.blocks
        if features_by_block_id.get(block.block_id) and features_by_block_id[block.block_id].recap_of_block_id
    }
    raw_tags = llm_service.call_json(
        prompt_name=prompts.PROMPT_NAME,
        prompt_version=prompts.PROMPT_VERSION,
        model=model_settings.tagging_model,
        effort="medium",
        system_prompt=prompts.SYSTEM_PROMPT,
        user_prompt=prompts.build_user_prompt(
            episode, show_title, bible_text, find_episode_summary(bible, episode.episode_number - 1), recap_block_ids
        ),
        output_model=EpisodeTags,
        max_output_tokens=_TAGGING_MAX_OUTPUT_TOKENS,
    )
    if raw_tags is None:
        logger.error("Episode %d is UNTAGGED; it falls back to emotion and sound signals", episode.episode_number)
        return EpisodeTags(episode_number=episode.episode_number, summary="", hooks=[], cliffhangers=[], arc_lines=[], is_untagged=True)
    return validate_episode_tags(raw_tags, episode, recap_block_ids)


def validate_episode_tags(raw_tags: EpisodeTags, episode: Episode, recap_block_ids: set[str]) -> EpisodeTags:
    """Drop tags that point at unknown or recap lines; fix answered_at values that cannot be right."""
    blocks_by_id = {block.block_id: block for block in episode.blocks}
    notes: list[str] = []
    hooks = []
    for hook in sorted(raw_tags.hooks, key=lambda tag: -tag.strength):
        if hook.block_id not in blocks_by_id or hook.block_id in recap_block_ids:
            notes.append(f"hook {hook.block_id} dropped: unknown or recap line")
            continue
        trigger = blocks_by_id.get(hook.trigger_block_id or "")
        if hook.trigger_block_id and (trigger is None or trigger.start_s > blocks_by_id[hook.block_id].start_s):
            notes.append(f"hook {hook.block_id}: trigger {hook.trigger_block_id} is not an earlier line; ignored")
            hook = hook.model_copy(update={"trigger_block_id": None})
        hooks.append(hook)
    cliffhangers = []
    for cliffhanger in sorted(raw_tags.cliffhangers, key=lambda tag: -tag.strength):
        if cliffhanger.block_id not in blocks_by_id or cliffhanger.block_id in recap_block_ids:
            notes.append(f"cliffhanger {cliffhanger.block_id} dropped: unknown or recap line")
            continue
        cliffhangers.append(_check_answered_at(cliffhanger, blocks_by_id, notes))
    arc_lines = [arc_line for arc_line in raw_tags.arc_lines if arc_line.block_id in blocks_by_id]
    if len(arc_lines) < len(raw_tags.arc_lines):
        notes.append(f"{len(raw_tags.arc_lines) - len(arc_lines)} arc lines dropped: unknown block ids")
    for note in notes:
        logger.info("Episode %d tags: %s", episode.episode_number, note)
    return EpisodeTags(
        episode_number=episode.episode_number,
        summary=raw_tags.summary,
        hooks=hooks[: config.TAGGING_MAX_HOOKS],
        cliffhangers=cliffhangers[: config.TAGGING_MAX_CLIFFHANGERS],
        arc_lines=arc_lines,
        validation_notes=notes,
    )


def _check_answered_at(cliffhanger, blocks_by_id: dict, notes: list[str]):
    if cliffhanger.answered_at in (ANSWERED_LATER, ANSWERED_NEVER):
        return cliffhanger
    answer = blocks_by_id.get(cliffhanger.answered_at)
    cliff_block = blocks_by_id[cliffhanger.block_id]
    if answer is None or answer.start_s < cliff_block.start_s:
        notes.append(f"cliffhanger {cliffhanger.block_id}: answered_at {cliffhanger.answered_at} is unknown or earlier; set to LATER")
        return cliffhanger.model_copy(update={"answered_at": ANSWERED_LATER})
    return cliffhanger
