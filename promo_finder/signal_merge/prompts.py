PROMPT_NAME = "disputed_review"
PROMPT_VERSION = "v1"

SYSTEM_PROMPT = """You check promo moments in a vertical microdrama. Each line below scored high on the writer's \
emotion tags or a sound cue, but was not picked as a promo moment. Decide for each whether a 20-60 second promo \
could open on it (HOOK: stops a scrolling stranger within 3 seconds), end on it (CLIFFHANGER: leaves an open \
question, cut before the answer), or NEITHER. Loud is not the same as gripping: say NEITHER for routine shouting, \
ambience or a moment that only matters with context. Strength 5 = stops almost anyone, 3 = strong with one line \
of setup, 1 = mild. Return one review per block id given."""


def build_user_prompt(story_bible_text: str, context_snippets: list[tuple[str, str]]) -> str:
    rendered_snippets = "\n\n".join(f"### Line {block_id}\n{snippet}" for block_id, snippet in context_snippets)
    return f"Story bible:\n{story_bible_text}\n\nLines to check, each with the lines around it (the line itself is marked >>):\n\n{rendered_snippets}"
