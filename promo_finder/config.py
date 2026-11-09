"""Every threshold, weight and model setting of the pipeline. Change values here, nowhere else."""

import os
from dataclasses import dataclass

# --- Promo shape (from the brief) ---
PROMO_MIN_DURATION_S = 20.0
PROMO_MAX_DURATION_S = 60.0
PROMO_TARGET_DURATION_S = 38.0
HOOK_WINDOW_S = 3.0  # the hook must land within the first 3 seconds
MIN_SEGMENT_DURATION_S = 3.0  # shorter segments read as a jump cut
MAX_SEGMENTS = 3

# --- Cutting on shots ---
SHOT_SNAP_TOLERANCE_S = 1.5  # start at the shot start if the hook is at most this far into the shot
REACTION_SHOT_MAX_S = 2.5  # a following shot this short, with no dialogue, is kept as the reaction
MID_SHOT_END_PAD_S = 0.5  # pad after the cliffhanger line when the shot runs on too long
CONTINUATION_GAP_MAX_S = 1.5  # a line opening with "..." continues the same speaker's line that ended this recently
CROSS_EPISODE_REACH_S = 45.0  # how far from the episode break a stitched hook or cliffhanger may sit
MAX_RAW_SPAN_S = 150.0  # hook-to-cliffhanger spans longer than this are not worth compressing

# --- Episode structure ---
EPISODE_TAIL_S = 10.0
EPISODE_HEAD_S = 12.0
SCENE_END_WINDOW_S = 3.0
RECAP_SEARCH_WINDOW_S = 25.0  # recaps of the previous ending sit in the opening seconds
RECAP_PREVIOUS_LINES = 8
RECAP_SEQUENCE_RATIO_MIN = 0.55
RECAP_WORD_OVERLAP_MIN = 0.6
RECAP_SHARED_WORDS_MIN = 2

# --- Signal thresholds ---
HOOK_SCORE_MIN = 0.5
CLIFFHANGER_SCORE_MIN = 0.5
CORROBORATION_SIGNAL_MIN = 0.5
SIGNAL_WINDOW_BEFORE_S = 0.5
SIGNAL_WINDOW_AFTER_S = 1.5
DISPUTED_Z_MIN = 2.0
DISPUTED_REVIEW_LIMIT = 20
TAGGING_MAX_HOOKS = 5
TAGGING_MAX_CLIFFHANGERS = 3

# --- Candidate selection ---
# Assumption: the first 10 episodes are free in these apps. A viewer who installs from a promo sees them within
# minutes, so they get the full early-episode bonus and are never held back as spoilers.
FREE_EPISODE_COUNT = 10
LLM_STRENGTH_ALWAYS_ELIGIBLE = 4  # LLM-rated hooks/cliffhangers this strong reach the cold viewer even without a signal
DEDUP_OVERLAP_MAX = 0.5
JUDGED_CANDIDATE_LIMIT = 35
EARLY_EPISODE_FULL_BONUS_THROUGH = FREE_EPISODE_COUNT
EARLY_EPISODE_FLOOR_AT = 40
EARLY_EPISODE_FLOOR = 0.3

# --- Weights: line-level merge ---
HOOK_LLM_WEIGHT = 0.6
HOOK_SIGNAL_WEIGHT = 0.4
CLIFFHANGER_LLM_WEIGHT = 0.6
CLIFFHANGER_BOUNDARY_WEIGHT = 0.2
CLIFFHANGER_SIGNAL_WEIGHT = 0.2

# --- Weights: candidate prefilter ---
PREFILTER_HOOK_WEIGHT = 0.30
PREFILTER_CLIFFHANGER_WEIGHT = 0.30
PREFILTER_SHAPE_WEIGHT = 0.15
PREFILTER_DURATION_WEIGHT = 0.10
PREFILTER_EARLY_EPISODE_WEIGHT = 0.15
PREFILTER_EXTRA_SEGMENT_PENALTY = 0.05
PREFILTER_CONTEXT_FACT_PENALTY = 0.05

# --- Weights: final ranking ---
FINAL_APPEAL_WEIGHT = 0.30
FINAL_COMPREHENSION_WEIGHT = 0.25
FINAL_CLIFFHANGER_MATCH_WEIGHT = 0.15
FINAL_HOOK_WEIGHT = 0.15
FINAL_SHAPE_WEIGHT = 0.15
EARLY_EPISODE_FINAL_BASE = 0.8  # final *= base + (1 - base) * early_episode
DIVERSITY_RELEVANCE_WEIGHT = 0.7
DIVERSITY_SIMILARITY_WEIGHT = 0.3
WHY_LINE_LIMIT = 40

# --- Pairwise comparison (final ordering) ---
PAIRWISE_POOL_SIZE = 15  # the top promos by final score that meet each other in pairs
PAIRWISE_NEIGHBOUR_WINDOW = 3  # each promo meets the next 3 in score order: 39 pairs, 78 calls for a pool of 15
PAIRWISE_BLEND_WEIGHT = 0.6  # posting relevance = 0.6 * pairwise strength + 0.4 * final score
BRADLEY_TERRY_ITERATIONS = 200

# --- LLM calls ---
LLM_TIMEOUT_S = 300.0
LLM_MAX_TRANSIENT_RETRIES = 3
LLM_MAX_PARALLEL_CALLS = 4
DEFAULT_MODEL = "claude-opus-5-5"

# USD per million tokens (input, output), for the run log's cost estimate.
MODEL_PRICES_USD_PER_MILLION_TOKENS = {
    "claude-opus-5-5": (4.00, 20.00),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-haiku-5-5": (0.10, 0.50),
}


@dataclass(frozen=True)
class ModelSettings:
    provider: str
    bible_model: str
    tagging_model: str
    judge_model: str
    viewer_model: str
    openai_compatible_base_url: str | None
    openai_compatible_api_key: str | None


def build_model_settings() -> ModelSettings:
    return ModelSettings(
        provider=os.getenv("LLM_PROVIDER", "anthropic"),
        bible_model=os.getenv("BIBLE_MODEL", DEFAULT_MODEL),
        tagging_model=os.getenv("TAGGING_MODEL", DEFAULT_MODEL),
        judge_model=os.getenv("JUDGE_MODEL", DEFAULT_MODEL),
        viewer_model=os.getenv("VIEWER_MODEL", DEFAULT_MODEL),
        openai_compatible_base_url=os.getenv("OPENAI_COMPATIBLE_BASE_URL") or None,
        openai_compatible_api_key=os.getenv("OPENAI_COMPATIBLE_API_KEY") or None,
    )


@dataclass(frozen=True)
class RunOptions:
    """Switches for ablation runs; the defaults are the full pipeline."""

    is_emotion_signal_enabled: bool = True
    is_cold_viewer_enabled: bool = True
    is_pairwise_enabled: bool = True
    max_episodes: int | None = None
    first_episode: int | None = None  # with last_episode: tag and judge only this window of episodes
    last_episode: int | None = None
