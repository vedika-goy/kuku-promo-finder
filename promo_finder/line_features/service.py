"""Model-free per-line signals: emotion intensity from the writer's tags, sound cue types, position, recaps."""

import difflib
import re
import statistics

from promo_finder import config
from promo_finder.line_features import config as lexicon
from promo_finder.line_features.schemas import LexiconCoverage, LineFeatures, SoundType
from promo_finder.screenplay.schemas import Block, BlockKind, Episode, Show

_WORD = re.compile(r"[a-z']+")
_STOP_WORDS = {
    "the", "a", "an", "and", "or", "but", "of", "to", "in", "on", "at", "for", "with", "is", "are", "was",
    "were", "be", "been", "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them", "my",
    "your", "his", "its", "our", "their", "this", "that", "so", "as", "not", "no", "do", "did", "have", "has",
    "had", "will", "would", "can", "could", "what", "who", "how", "why", "if", "then", "than", "too", "very",
}
_RECAP_STEM_LENGTH = 5


def _build_word_start_pattern(stems: tuple[str, ...]) -> re.Pattern:
    return re.compile(r"\b(?:" + "|".join(stems) + ")", re.I)


_SOUND_TYPE_PATTERNS = [(sound_type, _build_word_start_pattern(stems)) for sound_type, stems in lexicon.SOUND_TYPE_STEMS]
_ACTION_HIGH_PATTERN = _build_word_start_pattern(lexicon.ACTION_HIGH_STEMS)


def build_line_features(show: Show) -> dict[str, LineFeatures]:
    features_by_block_id: dict[str, LineFeatures] = {}
    previous_episode: Episode | None = None
    for episode in show.episodes:
        recap_matches = find_recap_matches(episode, previous_episode)
        features_by_block_id.update(_build_episode_features(episode, recap_matches))
        previous_episode = episode
    return features_by_block_id


def score_emotion_tags(emotion_tags: list[str]) -> tuple[float | None, bool]:
    """Returns (intensity of the strongest known tag or None, has a tension tag)."""
    intensities = [_score_tag(tag) for tag in emotion_tags]
    known_intensities = [intensity for intensity in intensities if intensity is not None]
    has_tension_tag = any(stem in tag.lower() for tag in emotion_tags for stem in lexicon.TENSION_TAG_STEMS)
    return (max(known_intensities) if known_intensities else None), has_tension_tag


def classify_sound(text: str) -> SoundType:
    for sound_type, pattern in _SOUND_TYPE_PATTERNS:
        if pattern.search(text):
            return sound_type
    return SoundType.OTHER


def measure_lexicon_coverage(show: Show) -> LexiconCoverage:
    tagged_lines = 0
    known_lines = 0
    unknown_words: set[str] = set()
    for episode in show.episodes:
        for block in episode.blocks:
            if block.kind != BlockKind.DIALOGUE or not block.emotion_tags:
                continue
            tagged_lines += 1
            if score_emotion_tags(block.emotion_tags)[0] is not None:
                known_lines += 1
            unknown_words.update(tag.lower() for tag in block.emotion_tags if _score_tag(tag) is None)
    return LexiconCoverage(tagged_dialogue_lines=tagged_lines, lines_with_known_tag=known_lines, unknown_tag_words=sorted(unknown_words))


def find_recap_matches(episode: Episode, previous_episode: Episode | None) -> dict[str, str]:
    """Opening dialogue lines that restage the previous episode's last lines: {block_id: previous block_id}."""
    if previous_episode is None:
        return {}
    previous_lines = [block for block in previous_episode.blocks if block.kind == BlockKind.DIALOGUE][-config.RECAP_PREVIOUS_LINES:]
    opening_lines = [
        block for block in episode.blocks
        if block.kind == BlockKind.DIALOGUE and block.start_s <= config.RECAP_SEARCH_WINDOW_S
    ]
    matches = {}
    for opening_line in opening_lines:
        best_match = next((line for line in previous_lines if _is_restaged_line(opening_line.text, line.text)), None)
        if best_match:
            matches[opening_line.block_id] = best_match.block_id
    return matches


def compute_recap_end_s(episode: Episode, features_by_block_id: dict[str, LineFeatures]) -> float:
    """End of the opening stretch that restages the previous episode; 0 when there is none."""
    recap_ends = [
        block.end_s for block in episode.content_blocks
        if features_by_block_id.get(block.block_id) and features_by_block_id[block.block_id].recap_of_block_id
    ]
    return max(recap_ends, default=0.0)


def _build_episode_features(episode: Episode, recap_matches: dict[str, str]) -> dict[str, LineFeatures]:
    content_blocks = episode.content_blocks
    scored = [(block, *_score_block(block)) for block in content_blocks]
    intensities = [intensity for _, intensity, _, _ in scored]
    mean_intensity = statistics.fmean(intensities) if intensities else 0.0
    std_intensity = max(statistics.pstdev(intensities) if len(intensities) > 1 else 0.0, 0.05)
    scene_ends = {scene.scene_number: scene.end_s for scene in episode.scenes}
    features = {}
    for index, (block, intensity, has_tension_tag, is_tag_known) in enumerate(scored):
        previous_intensities = intensities[max(0, index - 3):index]
        baseline = statistics.fmean(previous_intensities) if previous_intensities else intensity
        sound_type = classify_sound(block.text) if block.kind == BlockKind.SOUND else None
        scene_end_s = scene_ends.get(block.scene_number, episode.duration_s)
        features[block.block_id] = LineFeatures(
            block_id=block.block_id,
            episode_number=episode.episode_number,
            intensity=round(intensity, 3),
            delta=round(intensity - baseline, 3),
            z_in_episode=round((intensity - mean_intensity) / std_intensity, 3),
            has_tension_tag=has_tension_tag,
            sound_type=sound_type,
            is_cliffhanger_cue=block.kind == BlockKind.SOUND and lexicon.CLIFFHANGER_CUE_WORD in block.text.lower(),
            is_episode_head=block.start_s < config.EPISODE_HEAD_S,
            is_episode_tail=block.end_s >= episode.duration_s - config.EPISODE_TAIL_S,
            is_scene_end=block.end_s >= scene_end_s - config.SCENE_END_WINDOW_S,
            recap_of_block_id=recap_matches.get(block.block_id),
            is_tag_known=is_tag_known,
        )
    return features


def _score_block(block: Block) -> tuple[float, bool, bool]:
    """Returns (intensity, has tension tag, is the tag known to the word list)."""
    if block.kind == BlockKind.DIALOGUE:
        tag_intensity, has_tension_tag = score_emotion_tags(block.emotion_tags)
        intensity = tag_intensity if tag_intensity is not None else lexicon.UNTAGGED_DIALOGUE_INTENSITY
        if "!" in block.text:
            intensity = min(1.0, intensity + lexicon.EXCLAMATION_BONUS)
        return intensity, has_tension_tag, tag_intensity is not None
    if block.kind == BlockKind.SOUND:
        return lexicon.SOUND_INTENSITY[classify_sound(block.text)], False, True
    is_high_action = bool(_ACTION_HIGH_PATTERN.search(block.text))
    return (lexicon.ACTION_HIGH_INTENSITY if is_high_action else lexicon.ACTION_DEFAULT_INTENSITY), False, True


def _score_tag(tag: str) -> float | None:
    lowered_tag = tag.lower()
    for tier_intensity, stems in lexicon.EMOTION_TAG_TIERS:
        if any(stem in lowered_tag for stem in stems):
            return tier_intensity
    return None


def _is_restaged_line(opening_text: str, previous_text: str) -> bool:
    if difflib.SequenceMatcher(None, opening_text.lower(), previous_text.lower()).ratio() >= config.RECAP_SEQUENCE_RATIO_MIN:
        return True
    opening_stems = _content_stems(opening_text)
    previous_stems = _content_stems(previous_text)
    shared_stems = opening_stems & previous_stems
    if len(shared_stems) < config.RECAP_SHARED_WORDS_MIN:
        return False
    return len(shared_stems) / min(len(opening_stems), len(previous_stems)) >= config.RECAP_WORD_OVERLAP_MIN


def _content_stems(text: str) -> set[str]:
    return {word[:_RECAP_STEM_LENGTH] for word in _WORD.findall(text.lower()) if word not in _STOP_WORDS and len(word) > 2}
