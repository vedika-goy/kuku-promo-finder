from promo_finder.line_features.schemas import SoundType
from promo_finder.line_features.service import build_line_features, classify_sound, find_recap_matches, score_emotion_tags
from promo_finder.screenplay.schemas import Show
from promo_finder.screenplay.service import parse_episode_text
from tests.fakes.screenplay_factory import build_episode_text, build_standard_episode_scenes


def test_emotion_tags_map_to_intensity_tiers():
    assert score_emotion_tags(["Furious", "Screaming"])[0] == 0.9
    assert score_emotion_tags(["Cold", "Commanding"])[0] == 0.55
    assert score_emotion_tags(["Gentle"])[0] == 0.25
    assert score_emotion_tags(["Zxqv"])[0] is None


def test_tension_tags_are_detected():
    assert score_emotion_tags(["Shocked", "Trembling"])[1] is True
    assert score_emotion_tags(["Warm"])[1] is False


def test_sound_cues_are_typed_and_crackle_is_not_an_impact():
    assert classify_sound("A sharp slap echoes.") == SoundType.IMPACT
    assert classify_sound("Ominous drone swells to a cliffhanger peak.") == SoundType.STING
    assert classify_sound("Warm crackle of embers in the hearth.") == SoundType.AMBIENT
    assert classify_sound("Door hinges groan.") == SoundType.ENTRANCE


def test_features_mark_spikes_tail_and_cliffhanger_cue():
    episode = parse_episode_text(build_episode_text(1, build_standard_episode_scenes(1)), "e1.txt")
    features = build_line_features(Show(title="t", episodes=[episode]))
    slap_line = next(block for block in episode.blocks if block.text == "How dare you!")
    final_line = next(block for block in episode.blocks if block.text == "Then who is my real father?")
    cue = next(block for block in episode.blocks if "cliffhanger" in block.text)
    assert features[slap_line.block_id].z_in_episode > 1.0
    assert features[final_line.block_id].is_episode_tail
    assert features[final_line.block_id].has_tension_tag
    assert features[cue.block_id].is_cliffhanger_cue
    assert all(feature.block_id in {block.block_id for block in episode.content_blocks} for feature in features.values())


def test_recap_lines_restaging_the_previous_ending_are_found():
    previous = parse_episode_text(build_episode_text(1, build_standard_episode_scenes(1)), "e1.txt")
    recap_scene = [("CAMERA", 0.0, 3.0, "Close-up on Juliana."), ("JULIANA (Shocked)", 0.2, 2.8, "Then who is my real father?")]
    recap_scene += [("CAMERA", 3.0, 30.0, "New scene."), ("MAID (Calm)", 10.0, 12.0, "Breakfast is served in the garden.")]
    episode = parse_episode_text(build_episode_text(2, [recap_scene]), "e2.txt")
    matches = find_recap_matches(episode, previous)
    assert len(matches) == 1
    assert episode.blocks[1].block_id in matches
