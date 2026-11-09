import logging

from promo_finder.config import ModelSettings
from promo_finder.llm.service import LlmService
from promo_finder.screenplay.schemas import Show
from promo_finder.story_bible import prompts
from promo_finder.story_bible.schemas import EpisodeSummaryEntry, StoryBible

logger = logging.getLogger(__name__)

_BIBLE_MAX_OUTPUT_TOKENS = 32000


def build_story_bible(show: Show, llm_service: LlmService, model_settings: ModelSettings) -> StoryBible:
    bible = llm_service.call_json(
        prompt_name=prompts.PROMPT_NAME,
        prompt_version=prompts.PROMPT_VERSION,
        model=model_settings.bible_model,
        effort="medium",
        system_prompt=prompts.SYSTEM_PROMPT,
        user_prompt=prompts.build_user_prompt(show),
        output_model=StoryBible,
        max_output_tokens=_BIBLE_MAX_OUTPUT_TOKENS,
    )
    if bible is None:
        logger.error("Story bible call failed; continuing with a logline-only bible")
        return build_fallback_bible(show)
    return _drop_unknown_episodes(bible, {episode.episode_number for episode in show.episodes})


def build_fallback_bible(show: Show) -> StoryBible:
    return StoryBible(
        premise=show.episodes[0].logline if show.episodes else "",
        characters=[],
        arcs=[],
        twists=[],
        episode_summaries=[EpisodeSummaryEntry(episode_number=episode.episode_number, summary=episode.logline) for episode in show.episodes],
        is_fallback=True,
    )


def render_bible_for_prompt(bible: StoryBible, through_episode: int | None = None) -> str:
    """Compact text form for other prompts. `through_episode` hides twists and summaries after it."""
    character_lines = [
        f"- {character.canonical_name}: {character.role}. {character.relationships}"
        + (f" Secret: {character.secret}" if character.secret else "")
        + " Labels: " + ", ".join(f"{label.label} (ep {label.first_episode}-{label.last_episode})" for label in character.script_labels)
        for character in bible.characters
    ]
    arc_lines = [f"- {arc.arc_id} {arc.name} (ep {arc.first_episode}-{arc.last_episode}): {arc.summary}" for arc in bible.arcs]
    twist_lines = [
        f"- ep {twist.episode_number} [{twist.spoiler_level}]: {twist.description}"
        for twist in bible.twists
        if through_episode is None or twist.episode_number <= through_episode
    ]
    return (
        f"Premise: {bible.premise}\n"
        "Characters:\n" + "\n".join(character_lines) + "\n"
        "Arcs:\n" + "\n".join(arc_lines) + "\n"
        "Twists:\n" + ("\n".join(twist_lines) or "- none")
    )


def find_episode_summary(bible: StoryBible, episode_number: int) -> str:
    return next((entry.summary for entry in bible.episode_summaries if entry.episode_number == episode_number), "")


def _drop_unknown_episodes(bible: StoryBible, episode_numbers: set[int]) -> StoryBible:
    summaries = [entry for entry in bible.episode_summaries if entry.episode_number in episode_numbers]
    twists = [twist for twist in bible.twists if twist.episode_number in episode_numbers]
    dropped_count = len(bible.episode_summaries) - len(summaries) + len(bible.twists) - len(twists)
    if dropped_count:
        logger.warning("Story bible referenced %d unknown episode numbers; dropped them", dropped_count)
    return bible.model_copy(update={"episode_summaries": summaries, "twists": twists})
