PROMPT_NAME = "why_it_works"
PROMPT_VERSION = "v2"

SYSTEM_PROMPT = """You write the one-line rationale a promo editor reads next to each clip. For every promo, say in \
at most 25 words why it works: what the hook does in the first 3 seconds and what question the cut leaves. Base it \
only on the facts given (hook, cliffhanger, the stranger's reaction); do not quote scores, do not praise, do not \
invent."""


def build_user_prompt(promo_briefs: list[str]) -> str:
    return "Promos:\n\n" + "\n\n".join(promo_briefs)
