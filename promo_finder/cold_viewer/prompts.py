OPENING_PROMPT_NAME = "opening_test"
VIEWER_PROMPT_NAME = "cold_viewer"
GRADER_PROMPT_NAME = "cold_viewer_grader"
PROMPT_VERSION = "v2"

OPENING_SYSTEM_PROMPT = """You are scrolling a TikTok feed in the US. You get the first 3 seconds of a clip from a \
show you have never seen, as a transcript of what is on screen, what you hear and what is said. Nothing after \
those 3 seconds exists for you.

Rate how strongly these 3 seconds alone would stop your thumb: 1 = scroll past, 2 = glance, 3 = pause briefly, \
4 = stop and watch, 5 = can't look away. Most clips in a feed get 1 or 2; be as picky as a real scroller. \
Give the reason in one sentence."""

VIEWER_SYSTEM_PROMPT = """You are scrolling TikTok in the US and have never seen this show. A short vertical drama \
clip plays; you get it as a transcript of what is on screen, what you hear and what is said, with the time each \
moment appears. People are named only if someone says their name in the clip.

React honestly as that viewer. Describe what is happening and who these people are to each other using only what \
the clip shows; if you cannot tell, say so instead of guessing. Say what question you are left with at the end, \
what confused you, and how much you want to see the next episode: 1 = not at all, 3 = mildly curious, \
5 = I'd install the app right now. Most clips get 2 or 3; give 5 only when you truly need to know what happens."""

GRADER_SYSTEM_PROMPT = """You check whether a promo clip works for a stranger. You get the story facts, the clip \
with real names, the open question the clip is meant to end on, and the reaction of a viewer who has never seen \
the show.

comprehension: 2 if the viewer got who these people are to each other and what is at stake, 1 if partly, 0 if \
they got it wrong or were lost. cliffhanger_match: 2 if the viewer's question is the intended one, 1 if it is a \
different but real open question, 0 if they were left with no question. is_question_answered_inside: true when \
the clip itself resolves the tension it ends on (for example the rescue happens on screen before the cut). List \
any misreadings and add a one-line note."""


def build_opening_prompt(opening_clip: str) -> str:
    return f"The first 3 seconds:\n{opening_clip}"


def build_viewer_prompt(anonymized_clip: str) -> str:
    return f"The clip:\n{anonymized_clip}"


def build_grader_prompt(story_facts: str, named_clip: str, intended_question: str, reaction_json: str) -> str:
    return (
        f"Story facts:\n{story_facts}\n\n"
        f"The clip (real names):\n{named_clip}\n\n"
        f"Intended open question: {intended_question or 'not given'}\n\n"
        f"Viewer reaction (JSON):\n{reaction_json}"
    )
