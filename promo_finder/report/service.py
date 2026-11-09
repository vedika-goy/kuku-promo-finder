"""Write the editor-facing outputs: promos.json, promos.csv, report.md and emotion-curve plots."""

import csv
import json
from collections import Counter
from pathlib import Path

# The object-oriented Figure API needs no GUI backend, unlike pyplot.
from matplotlib.figure import Figure

from promo_finder import config
from promo_finder.candidates.schemas import Candidate, CandidateSet
from promo_finder.ranking.schemas import RankedPromo
from promo_finder.screenplay.schemas import Show
from promo_finder.signal_merge.schemas import ScoredLine

_REPORT_TOP_COUNT = 10
_PLOT_COUNT = 10


def write_promo_outputs(output_dir: Path, ranked_promos: list[RankedPromo]) -> None:
    (output_dir / "promos.json").write_text(json.dumps([promo.model_dump() for promo in ranked_promos], indent=1), encoding="utf-8")
    with (output_dir / "promos.csv").open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["rank", "promo_id", "kind", "segments", "duration_s", "hook_at", "hook", "cliffhanger_at", "cliffhanger", "open_question", "why_it_works", "final_score", "flags"])
        for promo in ranked_promos:
            writer.writerow([
                promo.rank, promo.promo_id, promo.kind, _format_segments(promo), promo.duration_s,
                f"Ep {promo.hook.episode_number} {promo.hook.at}", promo.hook.text,
                f"Ep {promo.cliffhanger.episode_number} {promo.cliffhanger.at}", promo.cliffhanger.text,
                promo.open_question, promo.why_it_works, promo.final_score, " ".join(promo.flags),
            ])


def write_report(
    output_dir: Path, show: Show, ranked_promos: list[RankedPromo], candidate_set: CandidateSet, post_judge_rejections: list[Candidate], run_summary: dict
) -> None:
    rejection_counts = Counter(reason.value for candidate in [*candidate_set.rejected, *post_judge_rejections] for reason in candidate.rejection_reasons)
    lines = [
        f"# Promo report: {show.title}",
        "",
        f"Episodes parsed: {len(show.episodes)}. Promos ranked: {len(ranked_promos)}. "
        f"Candidates built: {run_summary['candidates_built']}; passed the rules and dedup: {len(candidate_set.kept)}; "
        f"held for spoilers: {len(candidate_set.held_for_spoilers)}; judged by the cold viewer: {run_summary['candidates_judged']}.",
        "",
        f"## Top {_REPORT_TOP_COUNT}",
        "",
        "| # | Segments | Len | Hook | Cliffhanger | Why it works | Score |",
        "|---|---|---|---|---|---|---|",
    ]
    for promo in ranked_promos[:_REPORT_TOP_COUNT]:
        lines.append(
            f"| {promo.rank} | {_format_segments(promo)} | {promo.duration_s:.0f}s | {_escape(promo.hook.text)} | "
            f"{_escape(promo.cliffhanger.text)} | {_escape(promo.why_it_works)} | {promo.final_score:.3f} |"
        )
    lines += ["", "## Held for spoilers", ""]
    lines += [f"- {candidate.candidate_id} (Ep {candidate.segments[0].episode_number}, {candidate.duration_s:.0f}s): {candidate.open_question}" for candidate in candidate_set.held_for_spoilers[:10]] or ["- none"]
    lines += ["", "## Why candidates were rejected", ""]
    lines += [f"- {reason}: {count}" for reason, count in rejection_counts.most_common()] or ["- none"]
    lines += ["", "## Run", "", "```json", json.dumps(run_summary, indent=1), "```", ""]
    lines += ["## Emotion curves (top promos)", ""] + [f"![promo {promo.rank}](emotion_curves/promo_{promo.rank:02d}.png)" for promo in ranked_promos[:_PLOT_COUNT]]
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_emotion_curves(output_dir: Path, ranked_promos: list[RankedPromo], line_map: dict[str, ScoredLine]) -> None:
    plot_dir = output_dir / "emotion_curves"
    plot_dir.mkdir(exist_ok=True)
    lines_by_episode: dict[int, list[ScoredLine]] = {}
    for line in line_map.values():
        if line.intensity is not None:
            lines_by_episode.setdefault(line.episode_number, []).append(line)
    for promo in ranked_promos[:_PLOT_COUNT]:
        _plot_promo_curve(plot_dir / f"promo_{promo.rank:02d}.png", promo, lines_by_episode)


def _plot_promo_curve(path: Path, promo: RankedPromo, lines_by_episode: dict[int, list[ScoredLine]]) -> None:
    times_s, intensities, elapsed_s = [], [], 0.0
    for segment in promo.segments:
        segment_lines = sorted(
            (line for line in lines_by_episode.get(segment.episode_number, []) if segment.start_s <= line.start_s < segment.end_s),
            key=lambda line: line.start_s,
        )
        times_s += [elapsed_s + line.start_s - segment.start_s for line in segment_lines]
        intensities += [line.intensity for line in segment_lines]
        elapsed_s += segment.end_s - segment.start_s
    figure = Figure(figsize=(6, 2.2))
    axis = figure.subplots()
    axis.axvspan(0, config.HOOK_WINDOW_S, color="#f5c4b3", alpha=0.6, label="hook window")
    axis.plot(times_s, intensities, marker="o", markersize=3, color="#534ab7")
    axis.set_xlim(0, max(elapsed_s, 1))
    axis.set_ylim(0, 1.05)
    axis.set_xlabel("seconds into promo")
    axis.set_ylabel("intensity")
    axis.set_title(f"#{promo.rank} {_format_segments(promo)}", fontsize=9)
    figure.tight_layout()
    figure.savefig(path, dpi=110)


def _format_segments(promo: RankedPromo) -> str:
    return " + ".join(f"Ep {segment.episode_number} {segment.start}-{segment.end}" for segment in promo.segments)


def _escape(text: str) -> str:
    return text.replace("|", "/").replace("\n", " ")

