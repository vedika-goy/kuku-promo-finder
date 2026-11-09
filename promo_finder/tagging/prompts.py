from promo_finder import config
from promo_finder.screenplay.schemas import Block, BlockKind, Episode

PROMPT_NAME = "episode_tagging"
PROMPT_VERSION = "v2"

SYSTEM_PROMPT = f"""You pick promo moments in a vertical microdrama. A promo is a 20-60 second clip posted on \
TikTok/Reels/Shorts to make strangers start the show: it opens on a hook that stops a scrolling viewer within \
3 seconds (a slap, a reveal, a shocking line) and ends on a cliffhanger, cut before the question is answered.

Mark the lines a promo could open on (hooks) and the lines it could end on (cliffhangers) in this episode, \
referring to lines only by the block ids given. Be selective: at most {config.TAGGING_MAX_HOOKS} hooks and \
{config.TAGGING_MAX_CLIFFHANGERS} cliffhangers, only moments you would actually post. SHOT lines describe what \
is on screen, so a hook or cliffhanger can be a SHOT, ACTION or SOUND line as well as dialogue.

Strength scale, the same for hooks and cliffhangers:
5 = stops almost anyone (a slap, an identity reveal, a life-or-death threat)
3 = strong inside the story but needs a line of setup
1 = mild

For each cliffhanger, give the open question a viewer is left with and the block id of the line that answers \
it in this episode, or LATER if it is answered in a later episode. Also mark the lines that carry each \
storyline (arc ids from the story bible) as ESSENTIAL (the clip makes no sense without it) or SUPPORTING; \
leave filler lines out. needs_context lists facts a stranger must already know for the hook to land.

spoiler_level is about what a stranger would lose by seeing the moment in an ad. The show's title and the premise \
set up in the opening episodes are known before anyone taps, so a moment that shows what the title promises is \
NONE. MAJOR is only for a reveal that settles the central mystery or a late turning point."""


def build_user_prompt(episode: Episode, show_title: str, story_bible_text: str, previous_summary: str, recap_block_ids: set[str]) -> str:
    rendered_lines = [_render_block(block, block.block_id in recap_block_ids) for block in episode.blocks]
    return (
        f"Show title: {show_title}\n\n"
        f"Story bible:\n{story_bible_text}\n\n"
        f"Previous episode: {previous_summary or 'none (first episode)'}\n\n"
        f"Episode {episode.episode_number}. Logline: {episode.logline}\n"
        "Lines (block id | kind | speaker (emotion) | text). Lines marked RECAP restage the previous episode's "
        "ending and must not be tagged.\n" + "\n".join(rendered_lines)
    )


def _render_block(block: Block, is_recap: bool) -> str:
    kind_label = "SHOT" if block.kind == BlockKind.CAMERA else block.kind.value.upper()
    recap_marker = " RECAP" if is_recap else ""
    if block.kind == BlockKind.DIALOGUE:
        emotion = f" ({', '.join(block.emotion_tags)})" if block.emotion_tags else ""
        return f"{block.block_id} | {kind_label}{recap_marker} | {block.speaker}{emotion} | \"{block.text}\""
    return f"{block.block_id} | {kind_label}{recap_marker} | {block.text}"
