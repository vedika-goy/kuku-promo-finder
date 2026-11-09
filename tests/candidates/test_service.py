from promo_finder import config
from promo_finder.candidates.schemas import CandidateKind, RejectionReason
from promo_finder.candidates.service import build_candidates, score_early_episode
from promo_finder.screenplay.schemas import Show
from promo_finder.screenplay.service import parse_episode_text
from promo_finder.story_bible.schemas import SpoilerLevel
from promo_finder.tagging.schemas import EpisodeTags
from tests.fakes.pipeline_factory import build_cliffhanger_tag, build_hook_tag, build_line_map, find_block_id
from tests.fakes.screenplay_factory import build_episode_text, build_standard_episode_scenes


def _tags(episode_number: int, hook_ids: list[str], cliffhanger_tags) -> EpisodeTags:
    return EpisodeTags(
        episode_number=episode_number, summary="", hooks=[build_hook_tag(block_id) for block_id in hook_ids], cliffhangers=cliffhanger_tags, arc_lines=[]
    )


def test_single_stretch_snaps_to_shots_and_keeps_the_reaction_shot(tmp_path):
    episode = parse_episode_text(build_episode_text(1, build_standard_episode_scenes(1)), "e1.txt")
    show = Show(title="t", episodes=[episode])
    hook_id, cliffhanger_id = find_block_id(episode, "How dare you!"), find_block_id(episode, "Then who is my real father?")
    line_map, features = build_line_map(show, {1: _tags(1, [hook_id], [build_cliffhanger_tag(cliffhanger_id)])}, tmp_path)
    candidate_set = build_candidates(show, line_map, features)
    promo = next(candidate for candidate in candidate_set.kept if candidate.hook_block_id == hook_id)
    assert promo.kind == CandidateKind.SINGLE
    assert promo.segments[0].start_s == 20.0  # the shot where the slap starts
    assert promo.segments[-1].end_s == 69.0  # the frozen reaction shot is kept, the black frame is not
    assert config.PROMO_MIN_DURATION_S <= promo.duration_s <= config.PROMO_MAX_DURATION_S
    assert promo.hook_offset_s <= config.HOOK_WINDOW_S
    assert promo.rejection_reasons == []


def test_clip_never_includes_the_answer_line(tmp_path):
    scenes = build_standard_episode_scenes(1)
    scenes[0] = [block for block in scenes[0] if block[3] not in ("Cut to black.",)]
    scenes[0] += [("CAMERA", 69.0, 72.0, "Clarissa leans in."), ("CLARISSA (Cold)", 69.2, 71.5, "The Duke is your father.")]
    episode = parse_episode_text(build_episode_text(1, scenes), "e1.txt")
    show = Show(title="t", episodes=[episode])
    hook_id = find_block_id(episode, "How dare you!")
    cliffhanger_id = find_block_id(episode, "Then who is my real father?")
    answer_id = find_block_id(episode, "The Duke is your father.")
    line_map, features = build_line_map(show, {1: _tags(1, [hook_id], [build_cliffhanger_tag(cliffhanger_id, answered_at=answer_id)])}, tmp_path)
    promo = next(candidate for candidate in build_candidates(show, line_map, features).kept if candidate.cliffhanger_block_id == cliffhanger_id)
    assert answer_id not in promo.content_block_ids
    assert promo.segments[-1].end_s <= 69.2


def test_long_span_is_compressed_without_dropping_essential_lines(tmp_path):
    blocks = [("CAMERA", 0.0, 3.0, "Close-up as Clarissa raises her hand."), ("CLARISSA (Furious)", 0.2, 2.5, "How dare you!")]
    for index in range(20):
        start_s = 3.0 + index * 4.0
        blocks += [("CAMERA", start_s, start_s + 4.0, f"Filler shot {index}."), ("MAID (Calm)", start_s + 0.5, start_s + 3.0, f"Filler line {index}.")]
    blocks += [("CAMERA", 83.0, 87.0, "Close-up on Juliana."), ("JULIANA (Shocked)", 83.5, 86.5, "Then who is my real father?")]
    episode = parse_episode_text(build_episode_text(1, [blocks]), "e1.txt")
    show = Show(title="t", episodes=[episode])
    hook_id, cliffhanger_id = find_block_id(episode, "How dare you!"), find_block_id(episode, "Then who is my real father?")
    line_map, features = build_line_map(show, {1: _tags(1, [hook_id], [build_cliffhanger_tag(cliffhanger_id)])}, tmp_path)
    candidate_set = build_candidates(show, line_map, features)
    promo = next(candidate for candidate in candidate_set.kept if candidate.hook_block_id == hook_id)
    assert promo.kind == CandidateKind.COMPRESSED
    assert promo.duration_s <= config.PROMO_MAX_DURATION_S
    assert len(promo.segments) <= config.MAX_SEGMENTS
    assert {hook_id, cliffhanger_id} <= set(promo.content_block_ids)
    assert promo.dropped_block_ids


def test_cross_episode_promo_skips_the_recap(tmp_path):
    first = parse_episode_text(build_episode_text(1, build_standard_episode_scenes(1)), "e1.txt")
    second_scene = [("CAMERA", 0.0, 3.0, "Close-up on Juliana."), ("JULIANA (Shocked)", 0.2, 2.8, "Then who is my real father?")]
    second_scene += [("CAMERA", 3.0, 8.0, "The doors burst open."), ("ACTION", 3.0, 8.0, "A stranger strides in.")]
    second_scene += [("CAMERA", 8.0, 12.0, "Close-up on the stranger."), ("STRANGER (Menacing)", 8.5, 11.5, "I have come for my daughter.")]
    second_scene += [("CAMERA", 12.0, 14.0, "Juliana gasps."), ("CAMERA", 14.0, 40.0, "Wide shot."), ("MAID (Calm)", 20.0, 22.0, "More tea?")]
    second = parse_episode_text(build_episode_text(2, [second_scene]), "e2.txt")
    show = Show(title="t", episodes=[first, second])
    hook_id = find_block_id(first, "Argument line 4.")
    cliffhanger_id = find_block_id(second, "I have come for my daughter.")
    tags = {1: _tags(1, [hook_id], []), 2: _tags(2, [], [build_cliffhanger_tag(cliffhanger_id)])}
    line_map, features = build_line_map(show, tags, tmp_path)
    candidate_set = build_candidates(show, line_map, features)
    promo = next(candidate for candidate in candidate_set.kept if candidate.kind == CandidateKind.CROSS_EPISODE)
    assert [segment.episode_number for segment in promo.segments] == [1, 2]
    assert promo.segments[1].start_s >= 3.0  # the restaged line at 0.2 s is skipped
    assert "RECAP_SKIPPED" in promo.flags


def test_overlapping_candidates_are_deduplicated(tmp_path):
    episode = parse_episode_text(build_episode_text(1, build_standard_episode_scenes(1)), "e1.txt")
    show = Show(title="t", episodes=[episode])
    hook_id = find_block_id(episode, "How dare you!")
    second_hook_id = find_block_id(episode, "Argument line 0.")
    cliffhanger_id = find_block_id(episode, "Then who is my real father?")
    tags = EpisodeTags(
        episode_number=1, summary="", arc_lines=[], cliffhangers=[build_cliffhanger_tag(cliffhanger_id)],
        hooks=[build_hook_tag(hook_id), build_hook_tag(second_hook_id)],
    )
    line_map, features = build_line_map(show, {1: tags}, tmp_path)
    candidate_set = build_candidates(show, line_map, features)
    assert len(candidate_set.kept) == 1
    assert any(RejectionReason.DUPLICATE in candidate.rejection_reasons for candidate in candidate_set.rejected)


def test_major_spoilers_are_held_only_after_the_free_episodes(tmp_path):
    for episode_number, is_held in ((1, False), (config.FREE_EPISODE_COUNT + 2, True)):
        episode = parse_episode_text(build_episode_text(episode_number, build_standard_episode_scenes(episode_number)), "e.txt")
        show = Show(title="t", episodes=[episode])
        tags = EpisodeTags(
            episode_number=episode_number, summary="", arc_lines=[],
            hooks=[build_hook_tag(find_block_id(episode, "How dare you!"), spoiler_level=SpoilerLevel.MAJOR)],
            cliffhangers=[build_cliffhanger_tag(find_block_id(episode, "Then who is my real father?"))],
        )
        line_map, features = build_line_map(show, {episode_number: tags}, tmp_path / str(episode_number))
        candidate_set = build_candidates(show, line_map, features)
        assert bool(candidate_set.held_for_spoilers) is is_held, episode_number
        assert bool(candidate_set.kept) is not is_held, episode_number


def test_llm_rated_four_without_a_signal_still_becomes_a_candidate(tmp_path):
    episode = parse_episode_text(build_episode_text(1, build_standard_episode_scenes(1)), "e1.txt")
    show = Show(title="t", episodes=[episode])
    quiet_hook_id = find_block_id(episode, "Argument line 0.")
    tags = EpisodeTags(
        episode_number=1, summary="", arc_lines=[], hooks=[build_hook_tag(quiet_hook_id, strength=4)],
        cliffhangers=[build_cliffhanger_tag(find_block_id(episode, "Then who is my real father?"))],
    )
    line_map, features = build_line_map(show, {1: tags}, tmp_path)
    assert line_map[quiet_hook_id].hook_score < config.HOOK_SCORE_MIN
    candidate_set = build_candidates(show, line_map, features)
    assert [candidate.hook_block_id for candidate in candidate_set.kept] == [quiet_hook_id]

    weak_tags = tags.model_copy(update={"hooks": [build_hook_tag(quiet_hook_id, strength=3)]})
    line_map, features = build_line_map(show, {1: weak_tags}, tmp_path)
    assert not build_candidates(show, line_map, features).kept


def test_early_episodes_score_higher():
    assert score_early_episode(3) == 1.0
    assert score_early_episode(25) < 1.0
    assert score_early_episode(48) == config.EARLY_EPISODE_FLOOR


def test_hook_that_continues_a_sentence_starts_where_the_sentence_starts(tmp_path):
    scenes = build_standard_episode_scenes(1)
    scenes[0] = [
        ("JULIANA (Defiant)", 27.0, 31.5, "...than beg for yours.") if block[3] == "Argument line 1." else block for block in scenes[0]
    ]
    episode = parse_episode_text(build_episode_text(1, scenes), "e1.txt")
    show = Show(title="t", episodes=[episode])
    continuation_id = find_block_id(episode, "...than beg for yours.")
    sentence_start_id = find_block_id(episode, "Argument line 0.")
    tags = _tags(1, [continuation_id], [build_cliffhanger_tag(find_block_id(episode, "Then who is my real father?"))])
    line_map, features = build_line_map(show, {1: tags}, tmp_path)
    line_map[continuation_id] = line_map[continuation_id].model_copy(update={"hook_score": 1.0})
    promo = next(candidate for candidate in build_candidates(show, line_map, features).kept if candidate.hook_block_id == continuation_id)
    assert promo.hook_start_block_id == sentence_start_id
    assert promo.segments[0].start_s == 23.0  # the shot where "Argument line 0." begins, not mid-sentence at 27.0
    assert "HOOK_SENTENCE_JOINED" in promo.flags

    from promo_finder.ranking.service import build_hook_text

    assert build_hook_text(promo, show.build_block_index()) == "Argument line 0. ...than beg for yours."


def test_clip_never_opens_on_the_tail_of_a_sentence(tmp_path):
    scenes = build_standard_episode_scenes(1)
    scenes[0] = [block for block in scenes[0] if block[3] != "Thank you, that will be all."]
    scenes[0] += [("LORD (Cold)", 15.0, 19.0, "Your father has chosen his family..."), ("LORD (Cold)", 19.2, 20.1, "...and it isn't you.")]
    episode = parse_episode_text(build_episode_text(1, scenes), "e1.txt")
    show = Show(title="t", episodes=[episode])
    hook_id = find_block_id(episode, "How dare you!")
    fragment_id = find_block_id(episode, "...and it isn't you.")
    hook_with_trigger = build_hook_tag(hook_id).model_copy(update={"trigger_block_id": fragment_id})
    tags = EpisodeTags(
        episode_number=1, summary="", arc_lines=[], hooks=[hook_with_trigger],
        cliffhangers=[build_cliffhanger_tag(find_block_id(episode, "Then who is my real father?"))],
    )
    line_map, features = build_line_map(show, {1: tags}, tmp_path)
    promo = next(candidate for candidate in build_candidates(show, line_map, features).kept if candidate.hook_block_id == hook_id)
    assert fragment_id not in promo.content_block_ids  # its sentence starts 5 s before the hook, so the fragment is cut
    assert promo.segments[0].start_s >= 20.1
