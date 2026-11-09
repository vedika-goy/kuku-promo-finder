from promo_finder.screenplay.schemas import BlockKind, Show

PROMPT_NAME = "story_bible"
PROMPT_VERSION = "v2"

SYSTEM_PROMPT = """You write the story bible for a vertical microdrama. Later steps use it to judge short promo clips, \
so it must let someone tell who each speaker is, what each storyline is, and which reveals would spoil the show.

Use only what the scripts say. The same person can appear under several speaker labels (for example a description \
before their name is revealed), and one label can mean different people in different episodes: list every label \
with the episode range where it means this character. Arc ids are short (A1, A2, ...). The show's title and \
the premise set up in the opening episodes are known to every viewer before they start, so they are not twists. \
Mark a twist MAJOR when knowing it would spoil the show's central mystery or a late turning point, MINOR when it \
is a smaller surprise. Episode summaries are one or two sentences."""


def build_user_prompt(show: Show) -> str:
    sections = [f"Show: {show.title}. {len(show.episodes)} episodes follow."]
    for episode in show.episodes:
        dialogue_lines = [
            f"{block.speaker}{_qualifier_suffix(block.speaker_qualifiers)}: {block.text}"
            for block in episode.blocks
            if block.kind == BlockKind.DIALOGUE
        ]
        sections.append(
            f"## Episode {episode.episode_number}\n"
            f"Logline: {episode.logline}\n"
            f"Continuity notes: {episode.continuity_notes}\n"
            "Dialogue:\n" + "\n".join(dialogue_lines)
        )
    return "\n\n".join(sections)


def _qualifier_suffix(qualifiers: list[str]) -> str:
    return "".join(f" ({qualifier})" for qualifier in qualifiers)
