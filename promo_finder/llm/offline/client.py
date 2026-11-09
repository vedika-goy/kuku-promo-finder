"""A deterministic stand-in for an LLM (LLM_PROVIDER=offline): smoke runs and tests without a key.

It answers each prompt with simple rules so every stage runs end to end. Its picks are not meant to be good.
"""

import json
import re

from promo_finder.llm.schemas import LlmCompletion, LlmJsonRequest, LlmUsage

_TAGGED_LINE = re.compile(r"^(E\d+-B\d+) \| (\w+)( RECAP)? \| (.*)$", re.M)
_EPISODE_HEADER = re.compile(r"^## Episode (\d+)\nLogline: (.*)$", re.M)
_PROMO_ID = re.compile(r"^promo_id: (\S+)$", re.M)
_SNIPPET_ID = re.compile(r"^### Line (E\d+-B\d+)$", re.M)
_OFFLINE_MAX_HOOKS = 5
_OFFLINE_MAX_CLIFFHANGERS = 3


class OfflineLlmClient:
    def complete_json(self, request: LlmJsonRequest) -> LlmCompletion:
        builders = {
            "story_bible": self._build_story_bible,
            "episode_tagging": self._build_episode_tags,
            "disputed_review": self._build_disputed_review,
            "opening_test": self._build_opening_reaction,
            "cold_viewer": self._build_viewer_reaction,
            "cold_viewer_grader": self._build_grader_verdict,
            "why_it_works": self._build_why_lines,
            "pairwise_comparison": self._build_pair_verdict,
        }
        if request.prompt_name not in builders:
            raise ValueError(f"Offline client has no rule for prompt '{request.prompt_name}'")
        reply = builders[request.prompt_name](request.user_prompt)
        return LlmCompletion(text=json.dumps(reply), usage=LlmUsage(), served_by_model="offline", stop_reason="end_turn")

    def _build_story_bible(self, user_prompt: str) -> dict:
        summaries = [{"episode_number": int(number), "summary": logline} for number, logline in _EPISODE_HEADER.findall(user_prompt)]
        premise = summaries[0]["summary"] if summaries else ""
        return {"premise": premise, "characters": [], "arcs": [], "twists": [], "episode_summaries": summaries, "is_fallback": False}

    def _build_episode_tags(self, user_prompt: str) -> dict:
        lines = [(block_id, kind, rest) for block_id, kind, recap, rest in _TAGGED_LINE.findall(user_prompt) if not recap]
        dialogue_lines = [(block_id, rest) for block_id, kind, rest in lines if kind == "DIALOGUE"]
        exclaimed = [block_id for block_id, rest in dialogue_lines if "!" in rest][:_OFFLINE_MAX_HOOKS]
        questions = [block_id for block_id, rest in dialogue_lines if rest.rstrip().endswith('?"')]
        closing = [dialogue_lines[-1][0]] if dialogue_lines else []
        cliffhanger_ids = list(dict.fromkeys(closing + questions))[:_OFFLINE_MAX_CLIFFHANGERS]
        episode_match = re.search(r"^Episode (\d+)\. Logline", user_prompt, re.M)
        return {
            "episode_number": int(episode_match.group(1)) if episode_match else 0,
            "summary": "offline summary",
            "hooks": [_offline_hook(block_id) for block_id in exclaimed],
            "cliffhangers": [_offline_cliffhanger(block_id) for block_id in cliffhanger_ids],
            "arc_lines": [],
            "is_untagged": False,
            "validation_notes": [],
        }

    def _build_disputed_review(self, user_prompt: str) -> dict:
        reviews = [
            {"block_id": block_id, "verdict": "NEITHER", "strength": 1, "hook_type": None, "cliffhanger_type": None, "open_question": ""}
            for block_id in _SNIPPET_ID.findall(user_prompt)
        ]
        return {"reviews": reviews}

    def _build_opening_reaction(self, user_prompt: str) -> dict:
        return {"stop_strength": 4 if "!" in user_prompt else 2, "reason": "offline rule: an exclamation in the opening"}

    def _build_viewer_reaction(self, user_prompt: str) -> dict:
        return {
            "what_is_happening": "offline viewer",
            "who_is_who": "unknown",
            "open_question": "What happens next?",
            "confusing_parts": [],
            "want_next_score": 3,
        }

    def _build_grader_verdict(self, user_prompt: str) -> dict:
        return {"comprehension": 1, "cliffhanger_match": 1, "is_question_answered_inside": False, "misreadings": [], "note": "offline grader"}

    def _build_pair_verdict(self, user_prompt: str) -> dict:
        promo_a, _, promo_b = user_prompt.partition("Promo B (")
        return {"winner": "A" if promo_a.count("!") >= promo_b.count("!") else "B", "reason": "offline rule: more exclamations"}

    def _build_why_lines(self, user_prompt: str) -> dict:
        return {"lines": [{"promo_id": promo_id, "why_it_works": "Offline run: no model judgment."} for promo_id in _PROMO_ID.findall(user_prompt)]}


def _offline_hook(block_id: str) -> dict:
    return {
        "block_id": block_id, "strength": 4, "hook_type": "SHOCK_LINE", "trigger_block_id": None,
        "needs_context": [], "spoiler_level": "NONE", "reason": "offline rule: exclaimed line",
    }


def _offline_cliffhanger(block_id: str) -> dict:
    return {"block_id": block_id, "strength": 4, "cliffhanger_type": "CUT_BEFORE_ANSWER", "open_question": "What happens next?", "answered_at": "LATER"}
