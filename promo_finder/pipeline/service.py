"""Run every stage in order; each stage writes its own file so it can be inspected on its own."""

import json
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel

from promo_finder import config
from promo_finder.candidates.service import build_candidates
from promo_finder.cold_viewer.schemas import JudgedCandidate
from promo_finder.cold_viewer.service import JudgeContext, judge_candidates
from promo_finder.config import ModelSettings, RunOptions
from promo_finder.line_features.service import build_line_features, measure_lexicon_coverage
from promo_finder.llm.cache.repository import LlmCacheRepository
from promo_finder.llm.service import LlmClient, LlmService, build_llm_client
from promo_finder.ranking.schemas import RankedPromo
from promo_finder.pairwise_comparison.schemas import PairwiseResult
from promo_finder.pairwise_comparison.service import compare_in_pairs
from promo_finder.ranking.service import build_ranked_promos, order_for_posting, score_final
from promo_finder.report.service import write_emotion_curves, write_promo_outputs, write_report
from promo_finder.screenplay.schemas import Show
from promo_finder.screenplay.service import load_show
from promo_finder.signal_merge.schemas import HookEvidence
from promo_finder.signal_merge.service import merge_signals
from promo_finder.story_bible.service import build_story_bible
from promo_finder.tagging.service import tag_episodes

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    output_dir: Path
    ranked_promos: list[RankedPromo]
    run_summary: dict


def run_pipeline(
    input_path: Path,
    output_root: Path,
    cache_dir: Path,
    model_settings: ModelSettings,
    run_options: RunOptions,
    llm_client: LlmClient | None = None,
) -> PipelineResult:
    started_at_s = time.monotonic()
    full_show = load_show(input_path)
    show = _select_episodes(full_show, run_options)
    # The bible and recap detection need the story so far, not just the window: a promo from Ep 40 is judged
    # knowing Ep 1-39, and a window's first episode is checked for a recap of the episode before it.
    story_so_far = full_show.model_copy(update={"episodes": [episode for episode in full_show.episodes if episode.episode_number <= show.episodes[-1].episode_number]})
    output_dir = output_root / _build_run_folder_name(show.title, run_options)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Stage 1/8 parse: %d episodes -> %s", len(show.episodes), output_dir)
    _write_json(output_dir / "01_lines.json", [episode.model_dump(mode="json") for episode in show.episodes])

    logger.info("Stage 2/8 line features")
    features_by_block_id = build_line_features(story_so_far)
    lexicon_coverage = measure_lexicon_coverage(show)
    _write_json(output_dir / "02_line_features.json", {block_id: feature.model_dump(mode="json") for block_id, feature in features_by_block_id.items()})

    llm_service = LlmService(llm_client or build_llm_client(model_settings), LlmCacheRepository(cache_dir))
    logger.info("Stage 3/8 story bible (%s)", model_settings.bible_model)
    bible = build_story_bible(story_so_far, llm_service, model_settings)
    _write_model(output_dir / "03_story_bible.json", bible)

    logger.info("Stage 4/8 tagging %d episodes (%s)", len(show.episodes), model_settings.tagging_model)
    tags_by_episode = tag_episodes(show, bible, features_by_block_id, llm_service, model_settings)
    tags_dir = output_dir / "04_llm_tags"
    tags_dir.mkdir(exist_ok=True)
    for episode_number, tags in tags_by_episode.items():
        _write_model(tags_dir / f"E{episode_number:02d}.json", tags)

    logger.info("Stage 5/8 merge signals")
    line_map = merge_signals(show, features_by_block_id, tags_by_episode, bible, llm_service, model_settings, run_options)
    _write_json(output_dir / "05_line_map.json", {block_id: line.model_dump(mode="json") for block_id, line in line_map.items()})

    logger.info("Stage 6/8 build candidates")
    candidate_set = build_candidates(show, line_map, features_by_block_id)
    _write_model(output_dir / "06_candidates.json", candidate_set)

    to_judge = candidate_set.kept[: config.JUDGED_CANDIDATE_LIMIT]
    logger.info("Stage 7/8 cold viewer on %d candidates (enabled=%s)", len(to_judge), run_options.is_cold_viewer_enabled)
    judge_context = JudgeContext(
        show=show, line_map=line_map, features_by_block_id=features_by_block_id, bible=bible, llm_service=llm_service, model_settings=model_settings
    )
    judged = judge_candidates(to_judge, judge_context) if run_options.is_cold_viewer_enabled else [JudgedCandidate(candidate=candidate) for candidate in to_judge]
    _write_json(output_dir / "07_judged.json", [judged_candidate.model_dump(mode="json") for judged_candidate in judged])
    survivors = [judged_candidate for judged_candidate in judged if not judged_candidate.candidate.rejection_reasons]
    post_judge_rejections = [judged_candidate.candidate for judged_candidate in judged if judged_candidate.candidate.rejection_reasons]

    pairwise_result = _compare_in_pairs(survivors, judge_context, run_options)
    _write_model(output_dir / "08_pairwise.json", pairwise_result)
    strengths = pairwise_result.strength_by_promo_id
    logger.info("Stage 8/8 rank %d promos", len(survivors))
    ordered = order_for_posting(survivors, run_options.is_cold_viewer_enabled, strengths)
    ranked_promos = build_ranked_promos(ordered, show, llm_service, model_settings, strengths)
    run_summary = _build_run_summary(
        input_path, show, model_settings, run_options, lexicon_coverage, tags_by_episode, line_map, candidate_set, judged, ranked_promos, llm_service, started_at_s
    )
    run_summary["pairwise"] = {
        "pairs_judged": len(pairwise_result.outcomes),
        "pairs_failed": pairwise_result.failed_pairs,
        "ties_from_order_disagreement": sum(1 for outcome in pairwise_result.outcomes if outcome.first_score == 0.5),
    }
    write_promo_outputs(output_dir, ranked_promos)
    write_emotion_curves(output_dir, ranked_promos, line_map)
    write_report(output_dir, show, ranked_promos, candidate_set, post_judge_rejections, run_summary)
    _write_json(output_dir / "run_log.json", {**run_summary, "llm_calls": [record.model_dump() for record in llm_service.call_records]})
    logger.info("Done: %d promos, est. cost $%.2f -> %s", len(ranked_promos), run_summary["llm_usage"]["estimated_cost_usd"], output_dir)
    return PipelineResult(output_dir=output_dir, ranked_promos=ranked_promos, run_summary=run_summary)


def _compare_in_pairs(survivors: list[JudgedCandidate], judge_context: JudgeContext, run_options: RunOptions) -> PairwiseResult:
    """Pairwise ordering needs the cold viewer's verdicts to pick the pool, so it is off when the viewer is off."""
    if not (run_options.is_cold_viewer_enabled and run_options.is_pairwise_enabled):
        return PairwiseResult(outcomes=[], strength_by_promo_id={}, failed_pairs=0)
    pool = sorted(survivors, key=lambda judged: -score_final(judged, is_cold_viewer_enabled=True))[: config.PAIRWISE_POOL_SIZE]
    logger.info("Stage 7b pairwise comparison of the top %d promos", len(pool))
    return compare_in_pairs(pool, judge_context)


def _build_run_summary(
    input_path, show, model_settings, run_options, lexicon_coverage, tags_by_episode, line_map, candidate_set, judged, ranked_promos, llm_service, started_at_s
) -> dict:
    kind_counts: dict[str, int] = {}
    for promo in ranked_promos:
        kind_counts[promo.kind] = kind_counts.get(promo.kind, 0) + 1
    return {
        "input": str(input_path),
        "show": show.title,
        "episodes": len(show.episodes),
        "models": model_settings.__dict__ | {"openai_compatible_api_key": None},
        "run_options": run_options.__dict__,
        "lexicon_coverage": round(lexicon_coverage.coverage_ratio, 3),
        "untagged_episodes": [number for number, tags in tags_by_episode.items() if tags.is_untagged],
        "hooks_tagged": sum(len(tags.hooks) for tags in tags_by_episode.values()),
        "cliffhangers_tagged": sum(len(tags.cliffhangers) for tags in tags_by_episode.values()),
        "signal_only_hooks_confirmed": sum(1 for line in line_map.values() if line.hook_evidence == HookEvidence.SIGNAL_ONLY),
        "candidates_built": len(candidate_set.kept) + len(candidate_set.held_for_spoilers) + len(candidate_set.rejected),
        "candidates_kept": len(candidate_set.kept),
        "candidates_held_for_spoilers": len(candidate_set.held_for_spoilers),
        "candidates_judged": len(judged),
        "promos_ranked": len(ranked_promos),
        "promos_by_kind": kind_counts,
        "llm_usage": llm_service.summarize_usage(),
        "wall_time_s": round(time.monotonic() - started_at_s, 1),
    }


def _build_run_folder_name(show_title: str, run_options: RunOptions) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", show_title.lower()).strip("-")[:60]
    suffixes = []
    if not run_options.is_emotion_signal_enabled:
        suffixes.append("no-emotion")
    if not run_options.is_cold_viewer_enabled:
        suffixes.append("no-cold-viewer")
    if run_options.is_cold_viewer_enabled and not run_options.is_pairwise_enabled:
        suffixes.append("no-pairwise")
    if run_options.first_episode is not None:
        suffixes.append(f"episodes-{run_options.first_episode:02d}-{run_options.last_episode:02d}")
    elif run_options.max_episodes:
        suffixes.append(f"first-{run_options.max_episodes}")
    return "-".join([slug, *suffixes])


def _select_episodes(show: Show, run_options: RunOptions) -> Show:
    """The episodes this run tags and judges: a window, the first N, or all."""
    episodes = show.episodes
    if run_options.first_episode is not None:
        episodes = [episode for episode in episodes if run_options.first_episode <= episode.episode_number <= run_options.last_episode]
    elif run_options.max_episodes:
        episodes = episodes[: run_options.max_episodes]
    if not episodes:
        raise ValueError(f"No episodes selected (window {run_options.first_episode}-{run_options.last_episode}, first {run_options.max_episodes})")
    return show.model_copy(update={"episodes": episodes})


def _write_model(path: Path, model: BaseModel) -> None:
    path.write_text(model.model_dump_json(indent=1), encoding="utf-8")


def _write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")
