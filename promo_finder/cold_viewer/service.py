"""Test each candidate on a simulated stranger, then grade the stranger's understanding against the story facts."""

import logging
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from promo_finder import config
from promo_finder.candidates.schemas import Candidate, RejectionReason
from promo_finder.candidates.service import restore_dropped_line
from promo_finder.cold_viewer import prompts
from promo_finder.cold_viewer.schemas import GraderVerdict, JudgedCandidate, OpeningReaction, ViewerReaction
from promo_finder.config import ModelSettings
from promo_finder.line_features.schemas import LineFeatures
from promo_finder.llm.service import LlmService
from promo_finder.screenplay.schemas import Block, BlockKind, Show
from promo_finder.signal_merge.schemas import HookEvidence, ScoredLine
from promo_finder.story_bible.schemas import StoryBible
from promo_finder.story_bible.service import find_episode_summary, render_bible_for_prompt

logger = logging.getLogger(__name__)

_TITLE_WORDS = {"Lady", "Lord", "Duke", "Duchess", "Dowager", "Sir", "The", "Of", "Mr", "Mrs", "Miss", "Father", "Mother", "King", "Queen", "Prince", "Princess", "Earl", "Count", "Baron", "Madam"}
_LOWERCASE_WORD = re.compile(r"\b[a-z][a-z']+\b")
_KIND_LABELS = {BlockKind.CAMERA: "On screen", BlockKind.ACTION: "Action", BlockKind.SOUND: "Sound"}
_JUDGE_MAX_OUTPUT_TOKENS = 8000


@dataclass
class JudgeContext:
    show: Show
    line_map: dict[str, ScoredLine]
    features_by_block_id: dict[str, LineFeatures]
    bible: StoryBible
    llm_service: LlmService
    model_settings: ModelSettings


def judge_candidates(candidates: list[Candidate], context: JudgeContext) -> list[JudgedCandidate]:
    with ThreadPoolExecutor(max_workers=config.LLM_MAX_PARALLEL_CALLS) as executor:
        return list(executor.map(lambda candidate: judge_candidate(candidate, context), candidates))


def judge_candidate(candidate: Candidate, context: JudgeContext) -> JudgedCandidate:
    judged = _run_viewer_and_grader(candidate, context)
    if judged.grader_verdict and judged.grader_verdict.comprehension == 0 and candidate.dropped_block_ids:
        repaired_candidate = restore_dropped_line(candidate, context.show, context.line_map, context.features_by_block_id)
        if repaired_candidate is not None:
            repaired = _run_viewer_and_grader(repaired_candidate, context)
            if repaired.grader_verdict and repaired.grader_verdict.comprehension > 0:
                judged = repaired.model_copy(update={"is_repaired": True})
    judged.opening_reaction = _run_opening_test(judged.candidate, context)
    _apply_verdict_rules(judged)
    return judged


def render_clip(candidate: Candidate, show: Show, bible: StoryBible, is_anonymized: bool, max_offset_s: float | None = None) -> str:
    """The clip as a viewer gets it. `max_offset_s` keeps only what starts before that many seconds into the promo."""
    clip_blocks = _collect_clip_blocks(candidate, show)
    if max_offset_s is not None:
        clip_blocks = [item for item, offset_s in zip(clip_blocks, _relative_offsets(candidate, clip_blocks)) if offset_s < max_offset_s]
    spoken_text = " ".join(block.text for _, block in clip_blocks if block.kind == BlockKind.DIALOGUE)
    hidden_name_owners = _find_hidden_name_owners(bible, show, clip_blocks, spoken_text) if is_anonymized else {}
    person_labels: dict[str, str] = {}
    lines = []
    previous_segment_index = 0
    for (segment_index, block), offset_s in zip(clip_blocks, _relative_offsets(candidate, clip_blocks)):
        if segment_index != previous_segment_index:
            lines.append("--- cut ---")
            previous_segment_index = segment_index
        lines.append(f"[{offset_s:4.1f}s] " + _render_block(block, hidden_name_owners, person_labels))
    return "\n".join(lines)


def _run_opening_test(candidate: Candidate, context: JudgeContext) -> OpeningReaction | None:
    # Why a separate call: a viewer who has read the whole clip rates its opening too kindly
    # (in the first runs every clip got "would stop scrolling").
    opening_clip = render_clip(candidate, context.show, context.bible, is_anonymized=True, max_offset_s=config.HOOK_WINDOW_S)
    return context.llm_service.call_json(
        prompt_name=prompts.OPENING_PROMPT_NAME, prompt_version=prompts.PROMPT_VERSION, model=context.model_settings.viewer_model,
        effort="low", system_prompt=prompts.OPENING_SYSTEM_PROMPT, user_prompt=prompts.build_opening_prompt(opening_clip),
        output_model=OpeningReaction, max_output_tokens=_JUDGE_MAX_OUTPUT_TOKENS,
    )


def _run_viewer_and_grader(candidate: Candidate, context: JudgeContext) -> JudgedCandidate:
    anonymized_clip = render_clip(candidate, context.show, context.bible, is_anonymized=True)
    reaction = context.llm_service.call_json(
        prompt_name=prompts.VIEWER_PROMPT_NAME, prompt_version=prompts.PROMPT_VERSION, model=context.model_settings.viewer_model,
        effort="low", system_prompt=prompts.VIEWER_SYSTEM_PROMPT, user_prompt=prompts.build_viewer_prompt(anonymized_clip),
        output_model=ViewerReaction, max_output_tokens=_JUDGE_MAX_OUTPUT_TOKENS,
    )
    if reaction is None:
        return JudgedCandidate(candidate=candidate, rendered_clip=anonymized_clip)
    named_clip = render_clip(candidate, context.show, context.bible, is_anonymized=False)
    verdict = context.llm_service.call_json(
        prompt_name=prompts.GRADER_PROMPT_NAME, prompt_version=prompts.PROMPT_VERSION, model=context.model_settings.judge_model,
        effort="medium", system_prompt=prompts.GRADER_SYSTEM_PROMPT,
        user_prompt=prompts.build_grader_prompt(_render_story_facts(candidate, context.bible), named_clip, candidate.open_question, reaction.model_dump_json(indent=1)),
        output_model=GraderVerdict, max_output_tokens=_JUDGE_MAX_OUTPUT_TOKENS,
    )
    return JudgedCandidate(candidate=candidate, viewer_reaction=reaction, grader_verdict=verdict, rendered_clip=anonymized_clip)


def _apply_verdict_rules(judged: JudgedCandidate) -> None:
    candidate = judged.candidate
    if judged.viewer_reaction is None or judged.grader_verdict is None or judged.opening_reaction is None:
        candidate.rejection_reasons.append(RejectionReason.JUDGE_FAILED)
        return
    verdict = judged.grader_verdict
    if verdict.is_question_answered_inside:
        candidate.rejection_reasons.append(RejectionReason.QUESTION_ANSWERED_INSIDE)
    if verdict.comprehension == 0:
        candidate.rejection_reasons.append(RejectionReason.COLD_VIEWER_CONFUSED)
    elif verdict.comprehension == 1 and candidate.hook_evidence == HookEvidence.CONTEXT_HOOK:
        candidate.flags.append("CONTEXT_RISK")
    if judged.is_repaired:
        candidate.flags.append("REPAIRED")


def _render_story_facts(candidate: Candidate, bible: StoryBible) -> str:
    episode_numbers = sorted({segment.episode_number for segment in candidate.segments})
    summaries = "\n".join(f"Episode {number}: {find_episode_summary(bible, number)}" for number in episode_numbers)
    return render_bible_for_prompt(bible, through_episode=episode_numbers[-1]) + "\nEpisodes in the clip:\n" + summaries


def _collect_clip_blocks(candidate: Candidate, show: Show) -> list[tuple[int, Block]]:
    clip_blocks = []
    for segment_index, segment in enumerate(candidate.segments):
        episode = show.find_episode(segment.episode_number)
        clip_blocks.extend(
            (segment_index, block) for block in episode.blocks if segment.start_s <= block.start_s < segment.end_s - 0.05
        )
    return clip_blocks


def _relative_offsets(candidate: Candidate, clip_blocks: list[tuple[int, Block]]) -> list[float]:
    segment_offsets_s = []
    elapsed_s = 0.0
    for segment in candidate.segments:
        segment_offsets_s.append(elapsed_s)
        elapsed_s += segment.duration_s
    return [segment_offsets_s[index] + block.start_s - candidate.segments[index].start_s for index, block in clip_blocks]


def _find_hidden_name_owners(bible: StoryBible, show: Show, clip_blocks: list[tuple[int, Block]], spoken_text: str) -> dict[str, str]:
    """{name word: character} for proper names nobody says in the clip.

    A word counts as a name only if it never appears in lowercase anywhere in the show, so a descriptive name like
    'Mysterious Bearded Man' stays readable. A surname shared by several characters cannot be pinned to one person
    and stays visible. Sorted so the same clip always renders the same way (the LLM cache depends on it).
    """
    lowercase_words = {word for episode in show.episodes for block in episode.blocks for word in _LOWERCASE_WORD.findall(block.text)}
    owners_by_word: dict[str, set[str]] = {}
    for character in bible.characters:
        for word in re.findall(r"[A-Z][a-z]+", character.canonical_name):
            if word not in _TITLE_WORDS and word.lower() not in lowercase_words:
                owners_by_word.setdefault(word, set()).add(character.canonical_name)
    if not bible.characters:
        for _, block in clip_blocks:
            speaker_word = (block.speaker or "").strip().title()
            if block.kind == BlockKind.DIALOGUE and speaker_word and " " not in speaker_word and speaker_word.lower() not in lowercase_words:
                owners_by_word.setdefault(speaker_word, set()).add(speaker_word)
    unique_owners = {word: next(iter(owners)) for word, owners in owners_by_word.items() if len(owners) == 1}
    spoken_characters = {owner for word, owner in unique_owners.items() if re.search(rf"\b{re.escape(word)}\b", spoken_text)}
    return {word: owner for word, owner in sorted(unique_owners.items()) if owner not in spoken_characters}


def _render_block(block: Block, hidden_name_owners: dict[str, str], person_labels: dict[str, str]) -> str:
    if block.kind != BlockKind.DIALOGUE:
        return f"{_KIND_LABELS[block.kind]}: {_hide_names(block.text, hidden_name_owners, person_labels)}"
    speaker_label = _display_speaker(block.speaker or "Someone", hidden_name_owners, person_labels)
    tone = f" ({', '.join(tag.lower() for tag in block.emotion_tags)})" if block.emotion_tags else ""
    delivery = " [off screen]" if any(qualifier.upper() in ("O.S.", "OFF-SCREEN") for qualifier in block.speaker_qualifiers) else ""
    delivery = " [voice over]" if any(qualifier.upper() == "V.O." for qualifier in block.speaker_qualifiers) else delivery
    return f"{speaker_label}{delivery}{tone}: \"{block.text}\""


def _display_speaker(speaker: str, hidden_name_owners: dict[str, str], person_labels: dict[str, str]) -> str:
    title_case_speaker = speaker.title()
    owner = next((hidden_name_owners[word] for word in title_case_speaker.split() if word in hidden_name_owners), None)
    return _person_label(owner, person_labels) if owner else title_case_speaker


def _hide_names(text: str, hidden_name_owners: dict[str, str], person_labels: dict[str, str]) -> str:
    for word in sorted(hidden_name_owners, key=lambda name_word: (-len(name_word), name_word)):
        owner = hidden_name_owners[word]
        text = re.sub(rf"\b{re.escape(word)}('s)?\b", lambda match: _person_label(owner, person_labels) + (match.group(1) or ""), text)
    return text


def _person_label(character: str, person_labels: dict[str, str]) -> str:
    """One label per character, lettered in order of first appearance."""
    if character not in person_labels:
        person_labels[character] = f"Person {chr(ord('A') + len(person_labels) % 26)}"
    return person_labels[character]
