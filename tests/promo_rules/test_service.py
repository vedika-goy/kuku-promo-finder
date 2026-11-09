from promo_finder.candidates.schemas import RejectionReason, Segment
from promo_finder.promo_rules.service import check_promo_rules
from promo_finder.screenplay.service import parse_episode_text
from tests.fakes.screenplay_factory import build_episode_text, build_standard_episode_scenes

EPISODE = parse_episode_text(build_episode_text(1, build_standard_episode_scenes(1)), "e1.txt")
BLOCKS_BY_EPISODE = {1: EPISODE.blocks}
HOOK = next(block for block in EPISODE.blocks if block.text == "How dare you!")
CLIFFHANGER = next(block for block in EPISODE.blocks if block.text == "Then who is my real father?")
LAST_ARGUMENT = next(block for block in EPISODE.blocks if block.text == "Argument line 7.")


def _check(segments: list[Segment], cliffhanger=CLIFFHANGER, answer=None) -> list[RejectionReason]:
    return check_promo_rules(segments, HOOK, cliffhanger, answer, BLOCKS_BY_EPISODE, set())


def test_valid_promo_passes():
    assert _check([Segment(episode_number=1, start_s=20.0, end_s=69.0)]) == []


def test_too_short_and_too_long():
    assert RejectionReason.TOO_SHORT in _check([Segment(episode_number=1, start_s=20.0, end_s=35.0)], cliffhanger=HOOK)
    assert RejectionReason.TOO_LONG in _check([Segment(episode_number=1, start_s=0.0, end_s=69.0)])


def test_hook_after_three_seconds_is_late():
    assert RejectionReason.HOOK_LATE in _check([Segment(episode_number=1, start_s=16.0, end_s=69.0)])


def test_dialogue_after_the_cliffhanger_is_rejected():
    reasons = _check([Segment(episode_number=1, start_s=20.0, end_s=69.0)], cliffhanger=LAST_ARGUMENT)
    assert RejectionReason.DIALOGUE_AFTER_CLIFFHANGER in reasons


def test_answer_line_inside_the_clip_is_rejected():
    answer = LAST_ARGUMENT.model_copy(update={"start_s": 67.5, "end_s": 68.5})
    reasons = check_promo_rules([Segment(episode_number=1, start_s=20.0, end_s=69.0)], HOOK, CLIFFHANGER, answer, BLOCKS_BY_EPISODE, set())
    assert RejectionReason.ANSWER_INSIDE in reasons


def test_cut_through_a_line_and_short_segments():
    assert RejectionReason.CUT_THROUGH_LINE in _check([Segment(episode_number=1, start_s=20.0, end_s=65.0)])
    segments = [Segment(episode_number=1, start_s=20.0, end_s=30.0), Segment(episode_number=1, start_s=40.0, end_s=41.0), Segment(episode_number=1, start_s=63.0, end_s=69.0)]
    assert RejectionReason.SEGMENT_TOO_SHORT in _check(segments)


def test_recap_lines_inside_are_rejected():
    reasons = check_promo_rules([Segment(episode_number=1, start_s=20.0, end_s=69.0)], HOOK, CLIFFHANGER, None, BLOCKS_BY_EPISODE, {LAST_ARGUMENT.block_id})
    assert RejectionReason.RECAP_INSIDE in reasons
