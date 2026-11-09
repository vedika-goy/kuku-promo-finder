from pathlib import Path

import pytest

from promo_finder.screenplay.schemas import BlockKind
from promo_finder.screenplay.service import load_show, parse_episode_text, parse_timestamp_s
from tests.fakes.screenplay_factory import build_episode_text, build_standard_episode_scenes

REAL_PACK = Path(__file__).resolve().parents[2] / "data" / "I Married a Broken Soldier. He was a Duke"


def test_parse_timestamp_handles_minutes_and_hours():
    assert parse_timestamp_s("01:12.50") == 72.5
    assert parse_timestamp_s("1:01:12.50") == 3672.5
    assert parse_timestamp_s("00:00.00") == 0.0


def test_dialogue_label_gives_speaker_tags_and_qualifiers():
    text = build_episode_text(3, [[("CAMERA", 0, 2, "Shot."), ("ALEXANDER (O.S.) (Pained, Selfless)", 0.5, 2.0, "Go!")]])
    episode = parse_episode_text(text, "episode-003.txt")
    dialogue = next(block for block in episode.blocks if block.kind == BlockKind.DIALOGUE)
    assert dialogue.speaker == "ALEXANDER"
    assert dialogue.speaker_qualifiers == ["O.S."]
    assert dialogue.emotion_tags == ["Pained", "Selfless"]
    assert dialogue.text == "Go!"


def test_uppercase_labels_and_crlf_parse_the_same_as_the_standard_format():
    scenes = build_standard_episode_scenes(4)
    standard = parse_episode_text(build_episode_text(4, scenes), "a.txt")
    variant = parse_episode_text(build_episode_text(4, scenes, is_uppercase_labels=True, line_ending="\r\n"), "b.txt")
    assert [block.model_dump(exclude={"block_id"}) for block in standard.blocks] == [block.model_dump(exclude={"block_id"}) for block in variant.blocks]
    assert variant.scenes[0].end_s == 70.0
    assert variant.parse_warnings == []


def test_episode_number_falls_back_to_file_name():
    text = build_episode_text(7, build_standard_episode_scenes(7)).replace("EPISODE NUMBER: Episode 7", "EPISODE NUMBER: unknown")
    assert parse_episode_text(text, "episode-012-Episode 99.txt").episode_number == 12


def test_block_ending_before_it_starts_is_fixed_and_reported():
    text = build_episode_text(1, [[("CAMERA", 5.0, 3.0, "Broken timing.")]])
    episode = parse_episode_text(text, "x.txt")
    assert episode.blocks[0].end_s == episode.blocks[0].start_s
    assert episode.parse_warnings


def test_duplicate_episode_numbers_are_rejected(tmp_path):
    for name in ("one.txt", "two.txt"):
        (tmp_path / name).write_text(build_episode_text(1, build_standard_episode_scenes(1)), encoding="utf-8")
    with pytest.raises(ValueError, match="appears twice"):
        load_show(tmp_path)


@pytest.mark.skipif(not REAL_PACK.exists(), reason="input pack not present")
def test_real_input_pack_parses_completely():
    show = load_show(REAL_PACK)
    assert len(show.episodes) == 50
    counts = {kind: sum(1 for episode in show.episodes for block in episode.blocks if block.kind == kind) for kind in BlockKind}
    assert counts == {BlockKind.CAMERA: 1754, BlockKind.ACTION: 372, BlockKind.SOUND: 114, BlockKind.DIALOGUE: 1487}
    assert all(block.end_s >= block.start_s for episode in show.episodes for block in episode.blocks)
    assert show.find_episode(6).scenes, "episode 6 uses uppercase labels"
