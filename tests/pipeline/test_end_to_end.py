"""The whole pipeline on a synthetic 'new show' with a different file layout, using the offline LLM stand-in."""

import json

from promo_finder.config import RunOptions, build_model_settings
from promo_finder.llm.offline.client import OfflineLlmClient
from promo_finder.pipeline.service import run_pipeline
from promo_finder.promo_rules.service import check_promo_rules
from promo_finder.screenplay.service import load_show
from tests.fakes.screenplay_factory import build_episode_text, build_standard_episode_scenes


def _write_new_show(folder):
    folder.mkdir()
    for episode_number in (1, 2, 3):
        text = build_episode_text(
            episode_number, build_standard_episode_scenes(episode_number), is_uppercase_labels=episode_number == 2, line_ending="\r\n"
        )
        (folder / f"Part {episode_number} final.txt").write_text(text, encoding="utf-8")


def test_new_show_runs_end_to_end_and_every_promo_passes_the_rules(tmp_path):
    show_folder = tmp_path / "A New Show"
    _write_new_show(show_folder)
    result = run_pipeline(show_folder, tmp_path / "out", tmp_path / "cache", build_model_settings(), RunOptions(), llm_client=OfflineLlmClient())

    for file_name in ("01_lines.json", "05_line_map.json", "06_candidates.json", "07_judged.json", "08_pairwise.json", "promos.json", "promos.csv", "report.md", "run_log.json"):
        assert (result.output_dir / file_name).exists(), file_name
    assert result.ranked_promos, "the slap-to-cliffhanger stretch should produce promos"

    show = load_show(show_folder)
    blocks_by_id = show.build_block_index()
    blocks_by_episode = {episode.episode_number: episode.blocks for episode in show.episodes}
    for promo in json.loads((result.output_dir / "promos.json").read_text()):
        from promo_finder.candidates.schemas import Segment

        segments = [Segment(episode_number=segment["episode_number"], start_s=segment["start_s"], end_s=segment["end_s"]) for segment in promo["segments"]]
        reasons = check_promo_rules(
            segments, blocks_by_id[promo["hook"]["block_id"]], blocks_by_id[promo["cliffhanger"]["block_id"]], None, blocks_by_episode, set()
        )
        assert reasons == [], (promo["promo_id"], reasons)


def test_ablation_runs_write_to_their_own_folder(tmp_path):
    show_folder = tmp_path / "A New Show"
    _write_new_show(show_folder)
    options = RunOptions(is_emotion_signal_enabled=False, is_cold_viewer_enabled=False)
    result = run_pipeline(show_folder, tmp_path / "out", tmp_path / "cache", build_model_settings(), options, llm_client=OfflineLlmClient())
    assert result.output_dir.name.endswith("no-emotion-no-cold-viewer")


def test_episode_window_tags_only_the_window_but_the_bible_reads_the_story_so_far(tmp_path):
    show_folder = tmp_path / "A New Show"
    _write_new_show(show_folder)
    options = RunOptions(first_episode=2, last_episode=3)
    result = run_pipeline(show_folder, tmp_path / "out", tmp_path / "cache", build_model_settings(), options, llm_client=OfflineLlmClient())
    assert result.output_dir.name.endswith("episodes-02-03")
    assert [episode["episode_number"] for episode in json.loads((result.output_dir / "01_lines.json").read_text())] == [2, 3]
    bible = json.loads((result.output_dir / "03_story_bible.json").read_text())
    assert [entry["episode_number"] for entry in bible["episode_summaries"]] == [1, 2, 3]
    assert all(segment["episode_number"] in (2, 3) for promo in result.ranked_promos for segment in promo.model_dump()["segments"])
