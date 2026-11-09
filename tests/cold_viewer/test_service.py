from promo_finder.candidates.service import build_candidates
from promo_finder.cold_viewer.service import render_clip
from promo_finder.screenplay.schemas import Show
from promo_finder.screenplay.service import parse_episode_text
from promo_finder.story_bible.schemas import CharacterEntry, ScriptLabel, StoryBible
from promo_finder.tagging.schemas import EpisodeTags
from tests.fakes.pipeline_factory import build_cliffhanger_tag, build_hook_tag, build_line_map, find_block_id
from tests.fakes.screenplay_factory import build_episode_text, build_standard_episode_scenes

BIBLE = StoryBible(
    premise="A jilted bride.",
    characters=[
        CharacterEntry(canonical_name="Clarissa", script_labels=[ScriptLabel(label="CLARISSA", first_episode=1, last_episode=50)], role="sister", relationships="", secret=""),
        CharacterEntry(canonical_name="Lady Juliana", script_labels=[ScriptLabel(label="JULIANA", first_episode=1, last_episode=50)], role="heroine", relationships="", secret=""),
    ],
    arcs=[], twists=[], episode_summaries=[],
)


def _build_promo(tmp_path):
    episode = parse_episode_text(build_episode_text(1, build_standard_episode_scenes(1)), "e1.txt")
    show = Show(title="t", episodes=[episode])
    tags = EpisodeTags(
        episode_number=1, summary="", arc_lines=[], hooks=[build_hook_tag(find_block_id(episode, "How dare you!"))],
        cliffhangers=[build_cliffhanger_tag(find_block_id(episode, "Then who is my real father?"))],
    )
    line_map, features = build_line_map(show, {1: tags}, tmp_path)
    return show, build_candidates(show, line_map, features).kept[0]


def test_opening_render_stops_at_three_seconds(tmp_path):
    show, promo = _build_promo(tmp_path)
    opening = render_clip(promo, show, BIBLE, is_anonymized=True, max_offset_s=3.0)
    full = render_clip(promo, show, BIBLE, is_anonymized=True)
    assert "slaps" in opening and "How dare you!" in opening
    assert "Argument line 0." not in opening and "Argument line 0." in full


def test_unspoken_names_are_hidden_from_the_viewer_but_not_the_grader(tmp_path):
    show, promo = _build_promo(tmp_path)
    anonymized = render_clip(promo, show, BIBLE, is_anonymized=True)
    named = render_clip(promo, show, BIBLE, is_anonymized=False)
    assert "Clarissa" not in anonymized and "Juliana" not in anonymized
    assert "Person A" in anonymized
    assert "Clarissa" in named


def test_descriptive_names_stay_readable_and_one_character_gets_one_label(tmp_path):
    show, promo = _build_promo(tmp_path)
    descriptive = CharacterEntry(canonical_name="Wide Grand Hall", script_labels=[], role="", relationships="", secret="")
    bible = BIBLE.model_copy(update={"characters": [*BIBLE.characters, descriptive]})
    anonymized = render_clip(promo, show, bible, is_anonymized=True)
    assert "On screen: Close-up as Person A raises her hand." in anonymized
    assert "Action: Person A slaps Person B across the face." in anonymized  # same label for Clarissa every time
    assert "Person A (furious, screaming): \"How dare you!\"" in anonymized
    assert "Close-up on Person B, eyes wide." in anonymized  # 'wide' appears in lowercase in the show: not a name
    assert render_clip(promo, show, bible, is_anonymized=True) == anonymized
