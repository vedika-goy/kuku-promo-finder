"""Find promos in a show's screenplays.

    python run.py --input "<folder or .zip of episode .txt files>" --out outputs/
"""

import argparse
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from promo_finder.config import RunOptions, build_model_settings
from promo_finder.pipeline.service import run_pipeline

_PROJECT_DIR = Path(__file__).resolve().parent


def main() -> int:
    load_dotenv(_PROJECT_DIR / ".env")
    arguments = _parse_arguments()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    model_settings = build_model_settings()
    if arguments.provider:
        model_settings = model_settings.__class__(**{**model_settings.__dict__, "provider": arguments.provider})
    if model_settings.provider == "anthropic" and not (os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN")):
        print("No Anthropic credentials: add ANTHROPIC_API_KEY to .env (see .env.example), or run with --provider offline.", file=sys.stderr)
        return 2
    first_episode, last_episode = _parse_episode_window(arguments.episodes)
    run_options = RunOptions(
        first_episode=first_episode,
        last_episode=last_episode,
        is_emotion_signal_enabled=not arguments.no_emotion_signal,
        is_cold_viewer_enabled=not arguments.no_cold_viewer,
        is_pairwise_enabled=not arguments.no_pairwise,
        max_episodes=arguments.max_episodes,
    )
    result = run_pipeline(Path(arguments.input), Path(arguments.out), Path(arguments.cache_dir), model_settings, run_options)
    print(f"\n{len(result.ranked_promos)} promos -> {result.output_dir / 'report.md'}")
    for promo in result.ranked_promos[:5]:
        segments = " + ".join(f"Ep {segment.episode_number} {segment.start}-{segment.end}" for segment in promo.segments)
        print(f"#{promo.rank} {segments} ({promo.duration_s:.0f}s) hook: {promo.hook.text[:60]!r}")
    return 0


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Find 20-60 s promos in microdrama screenplays.")
    parser.add_argument("--input", required=True, help="Folder or .zip with one .txt screenplay per episode")
    parser.add_argument("--out", default=str(_PROJECT_DIR / "outputs"), help="Output root; a folder per show is created inside")
    parser.add_argument("--cache-dir", default=str(_PROJECT_DIR / ".cache" / "llm"), help="LLM reply cache (reruns are free)")
    parser.add_argument("--provider", choices=["anthropic", "openai_compatible", "offline"], help="Override LLM_PROVIDER from .env")
    parser.add_argument("--max-episodes", type=int, help="Only the first N episodes (quick checks)")
    parser.add_argument("--episodes", help="Only this window, e.g. 10-14 (the story bible still reads every earlier episode)")
    parser.add_argument("--no-emotion-signal", action="store_true", help="Ablation: LLM tags only, no emotion or sound signal")
    parser.add_argument("--no-cold-viewer", action="store_true", help="Ablation: rank by the prefilter score, skip the cold viewer")
    parser.add_argument("--no-pairwise", action="store_true", help="Ablation: order by the final score only, no head-to-head comparison")
    return parser.parse_args()


def _parse_episode_window(window: str | None) -> tuple[int | None, int | None]:
    if not window:
        return None, None
    first_text, _, last_text = window.partition("-")
    first_episode, last_episode = int(first_text), int(last_text or first_text)
    if first_episode > last_episode:
        raise SystemExit(f"--episodes {window}: the first episode must not be after the last")
    return first_episode, last_episode


if __name__ == "__main__":
    sys.exit(main())
