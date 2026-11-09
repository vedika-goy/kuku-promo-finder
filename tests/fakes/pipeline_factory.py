"""Shared setup for candidate tests: a parsed show, its features and a merged line map with chosen tags."""

from pathlib import Path

from promo_finder.config import RunOptions, build_model_settings
from promo_finder.line_features.service import build_line_features
from promo_finder.llm.cache.repository import LlmCacheRepository
from promo_finder.llm.offline.client import OfflineLlmClient
from promo_finder.llm.service import LlmService
from promo_finder.screenplay.schemas import Episode, Show
from promo_finder.signal_merge.service import merge_signals
from promo_finder.story_bible.service import build_fallback_bible
from promo_finder.story_bible.schemas import SpoilerLevel
from promo_finder.tagging.schemas import ANSWERED_LATER, CliffhangerTag, CliffhangerType, EpisodeTags, HookTag, HookType


def find_block_id(episode: Episode, text: str) -> str:
    return next(block.block_id for block in episode.blocks if block.text == text)


def build_hook_tag(block_id: str, strength: int = 5, spoiler_level: SpoilerLevel = SpoilerLevel.NONE) -> HookTag:
    return HookTag(
        block_id=block_id, strength=strength, hook_type=HookType.PHYSICAL, trigger_block_id=None,
        needs_context=[], spoiler_level=spoiler_level, reason="test",
    )


def build_cliffhanger_tag(block_id: str, answered_at: str = ANSWERED_LATER, strength: int = 5) -> CliffhangerTag:
    return CliffhangerTag(
        block_id=block_id, strength=strength, cliffhanger_type=CliffhangerType.REVEAL_PENDING,
        open_question="Who is her father?", answered_at=answered_at,
    )


def build_line_map(show: Show, tags_by_episode: dict[int, EpisodeTags], cache_dir: Path):
    features = build_line_features(show)
    for episode in show.episodes:
        tags_by_episode.setdefault(
            episode.episode_number, EpisodeTags(episode_number=episode.episode_number, summary="", hooks=[], cliffhangers=[], arc_lines=[])
        )
    llm_service = LlmService(OfflineLlmClient(), LlmCacheRepository(cache_dir))
    line_map = merge_signals(show, features, tags_by_episode, build_fallback_bible(show), llm_service, build_model_settings(), RunOptions())
    return line_map, features
