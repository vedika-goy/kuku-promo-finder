from promo_finder.screenplay.service import parse_episode_text
from promo_finder.tagging.schemas import ANSWERED_LATER, EpisodeTags
from promo_finder.tagging.service import validate_episode_tags
from tests.fakes.pipeline_factory import build_cliffhanger_tag, build_hook_tag, find_block_id
from tests.fakes.screenplay_factory import build_episode_text, build_standard_episode_scenes

EPISODE = parse_episode_text(build_episode_text(1, build_standard_episode_scenes(1)), "e1.txt")


def test_unknown_and_recap_block_ids_are_dropped():
    slap_id = find_block_id(EPISODE, "How dare you!")
    recap_id = find_block_id(EPISODE, "The tea is ready, episode 1.")
    raw = EpisodeTags(
        episode_number=1, summary="", arc_lines=[], cliffhangers=[],
        hooks=[build_hook_tag(slap_id), build_hook_tag("E01-B999"), build_hook_tag(recap_id)],
    )
    validated = validate_episode_tags(raw, EPISODE, {recap_id})
    assert [hook.block_id for hook in validated.hooks] == [slap_id]
    assert len(validated.validation_notes) == 2


def test_answer_before_the_cliffhanger_is_reset_to_later():
    cliffhanger_id = find_block_id(EPISODE, "Then who is my real father?")
    earlier_id = find_block_id(EPISODE, "How dare you!")
    raw = EpisodeTags(episode_number=1, summary="", arc_lines=[], hooks=[], cliffhangers=[build_cliffhanger_tag(cliffhanger_id, answered_at=earlier_id)])
    validated = validate_episode_tags(raw, EPISODE, set())
    assert validated.cliffhangers[0].answered_at == ANSWERED_LATER


def test_trigger_after_the_hook_is_ignored():
    hook = build_hook_tag(find_block_id(EPISODE, "How dare you!")).model_copy(update={"trigger_block_id": find_block_id(EPISODE, "Argument line 3.")})
    validated = validate_episode_tags(EpisodeTags(episode_number=1, summary="", arc_lines=[], cliffhangers=[], hooks=[hook]), EPISODE, set())
    assert validated.hooks[0].trigger_block_id is None
