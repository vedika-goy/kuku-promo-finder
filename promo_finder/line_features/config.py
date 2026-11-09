"""Word lists that turn the writer's emotion tags, action lines and sound cues into numbers.

Emotion-tag stems are matched as substrings of the lowercased tag, so 'desperat' covers desperate/desperately.
Tiers are checked from the highest down; the first tier with a match decides the tag's intensity.
Action and sound stems are regex fragments matched at the start of a word ('crack(?!l)' skips 'crackle').
"""

from promo_finder.line_features.schemas import SoundType

EMOTION_TAG_TIERS: list[tuple[float, tuple[str, ...]]] = [
    (0.9, (
        "desperat", "scream", "shout", "furious", "sob", "weep", "heartbroken", "hysteric", "terrif", "panic",
        "horrif", "outrag", "enrag", "devastat", "shatter", "agon", "anguish", "shriek", "frantic", "chok",
        "crying", "wail", "roar", "livid", "seeth", "screech", "yell", "despair", "beg", "venom", "vicious",
        "threaten", "menac", "sinister", "chilling", "deadly", "ruthless", "malicious", "vindictive", "vengeful",
        "lethal", "torment", "strangl", "broken", "betray", "blood-curdling",
    )),
    (0.75, (
        "defiant", "fierce", "mock", "taunt", "smug", "gloat", "scorn", "accus", "condemn", "contempt", "derisi",
        "sneer", "stunned", "shock", "alarm", "incredul", "tearful", "trembl", "breathless", "urgent", "sharp",
        "harsh", "bitter", "disgust", "indignant", "cutting", "scathing", "biting", "hiss", "snarl", "dangerous",
        "ominous", "warning", "forceful", "demanding", "intense", "passionate", "stricken", "distress", "fear",
        "frighten", "cruel", "spiteful", "pleading", "gasp", "triumph", "victorious", "plead", "pained", "hurt",
        "crushed", "overwhelm", "shaken", "shaking", "raw", "brutal", "aggressive", "heartless", "thunderous",
        "booming", "relentless", "goading", "provok", "coerc", "seductive", "aghast", "astonish", "astound",
        "barked", "barking", "damning", "agitated", "awestruck", "devious", "conniving",
    )),
    (0.55, (
        "cold", "stern", "firm", "resolute", "command", "authoritative", "steely", "grave", "solemn", "dismissive",
        "condescending", "arrogant", "haughty", "imperious", "calculat", "scheming", "proud", "determined",
        "unyielding", "steadfast", "unwavering", "decisive", "final", "powerful", "grim", "dark", "icy", "severe",
        "blunt", "confused", "hesitant", "anxious", "nervous", "suspicious", "wary", "tense", "strain", "worried",
        "uneasy", "apprehensive", "skeptical", "sarcastic", "sardonic", "vulnerable", "emotional", "resigned",
        "sorrow", "somber", "defeated", "haunted", "startl", "bewilder", "disbeliev", "unbeliev", "realiz",
        "insistent", "adamant", "uncompromising", "unforgiving", "imposing", "challenging", "interrogat",
        "scolding", "snide", "smirk", "annoyed", "impatient", "scoff", "condescend", "callous", "dignified",
        "unflinching", "unapologetic", "boastful", "conceited", "ambitious", "resentful", "indifferent",
        "dramatic", "theatrical", "vow", "sacrific", "praying", "exhausted", "weak", "faint", "faltering",
        "stiff", "guarded", "aloof", "denying", "pressing", "probing", "penetrating", "piercing", "loud",
        "hoarse", "panting", "muffled", "strangled", "hollow", "heavy", "aching", "melanchol", "guilt",
        "conflicted", "masking", "flustered", "submissive", "meek", "obedient", "deferential", "fragile",
    )),
    (0.4, ("whisper", "hushed", "murmur", "low", "quiet", "subdued", "muttering", "grated", "gruff", "deep", "dry")),
    (0.25, (
        "gentle", "soft", "calm", "sincere", "reassuring", "tender", "warm", "loving", "earnest", "humble",
        "polite", "formal", "measured", "steady", "reflective", "curious", "inquiring", "peaceful", "comfort",
        "caring", "fond", "grateful", "relieved", "pleased", "playful", "cheerful", "content", "matter-of-fact",
        "practical", "focused", "direct", "clear", "deliberate", "smooth", "amused", "intrigued", "impressed",
        "observant", "compassionate", "principled", "polite", "gracious", "nostalgic", "modest", "safe",
        "affirming", "simple", "honorable", "resolved", "tentative", "ceremonial", "procedural", "official",
        "instructional", "knowledgeable", "prompt", "committed", "transactional", "laughing", "chuckling",
        "giggling", "teasing", "sly", "conspiratorial", "promising", "fervent", "emphatic", "certain",
        "confident", "grounded", "devoted", "heartfelt", "reverent", "searching", "questioning", "wondering",
        "recognizing", "poised", "resilient", "apologetic", "unfazed", "unintimidated", "intent", "husky",
        "purring", "droll", "puzzled", "savoring", "intimate", "selfless", "fearless", "unshakable", "unbroken",
        "implacable", "cautious", "cautionary", "brooding", "weary", "sober", "definite", "absolute", "strict",
        "continu", "projecting", "resonant", "hurried", "observ", "transformed", "retreating", "comforted",
    )),
]

# Tags that signal unresolved tension: a strong sign the line can end a promo.
TENSION_TAG_STEMS = (
    "shock", "stun", "horrif", "terrif", "fear", "frighten", "alarm", "panic", "gasp", "dread", "ominous",
    "menac", "threat", "sinister", "chilling", "deadly", "dangerous", "warning", "suspicious", "startl",
    "incredul", "disbeliev", "unbeliev", "bewilder", "realiz", "vow", "lethal", "aghast", "astonish", "astound",
)

EXCLAMATION_BONUS = 0.05
UNTAGGED_DIALOGUE_INTENSITY = 0.5

ACTION_HIGH_INTENSITY = 0.85
ACTION_DEFAULT_INTENSITY = 0.35
ACTION_HIGH_STEMS = (
    "slap", "strike", "strikes", "punch", "kick", "shove", "grab", "seize", "yank", "drag", "throw", "hurl",
    "slam", "collaps", "faint", "kneel", "falls", "stab", "chok", "strangl", "kiss", "embrace", "scream", "sob",
    "weep", "gasp", "burst", "shatter", "smash", "tear", "rip", "crash", "lunge", "attack", "pins", "restrain",
    "bleed", "blood", "knife", "sword", "dagger", "blade", "poison", "pistol", "gun", "flame", "burn", "branding",
    "iron", "whip", "cower", "trembl", "freezes", "recoil", "stagger", "wrench",
)

# ORDER MATTERS: the first matching type wins, so a cue with both a door and a musical swell is a STING.
SOUND_TYPE_STEMS: list[tuple[SoundType, tuple[str, ...]]] = [
    (SoundType.IMPACT, (
        "slap", "punch", "strike", "crack(?!l)", "crash", "thud", "smash", "shatter", "slam", "thump", "clang", "bang",
        "gunshot", "whip", "crunch", "clatter", "sizzl", "searing", "blow",
    )),
    (SoundType.STING, ("chord", "sting", "swell", "drone", "cliffhanger", "crescendo", "dramatic", "ominous", "tension", "music", "orchestral", "thunder")),
    (SoundType.GASP, ("gasp", "scream", "shriek", "collective", "crowd", "cry", "murmur", "sob")),
    (SoundType.ENTRANCE, ("door", "footstep", "footfall", "hinge", "knock", "hooves", "carriage")),
    (SoundType.AMBIENT, ("rain", "wind", "crackl", "bird", "drip", "ambient", "rustl", "patter", "fire", "water")),
]

SOUND_INTENSITY = {
    SoundType.IMPACT: 0.9,
    SoundType.STING: 0.8,
    SoundType.GASP: 0.8,
    SoundType.ENTRANCE: 0.5,
    SoundType.AMBIENT: 0.2,
    SoundType.OTHER: 0.3,
}

CLIFFHANGER_CUE_WORD = "cliffhanger"
