from pydantic import BaseModel

from promo_finder.llm.anthropic.client import LlmRefusalError
from promo_finder.llm.cache.repository import LlmCacheRepository
from promo_finder.llm.schemas import LlmCompletion, LlmJsonRequest, LlmUsage
from promo_finder.llm.service import LlmService
from promo_finder.llm.utils import build_strict_json_schema


class _Answer(BaseModel):
    label: str
    score: int


class _ScriptedClient:
    """Returns the given replies in order; raises when a reply is an exception."""

    def __init__(self, replies: list) -> None:
        self.replies = list(replies)
        self.requests: list[LlmJsonRequest] = []

    def complete_json(self, request: LlmJsonRequest) -> LlmCompletion:
        self.requests.append(request)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return LlmCompletion(text=reply, usage=LlmUsage(input_tokens=10, output_tokens=5), served_by_model="test-model")


def _call(service: LlmService):
    return service.call_json("test_prompt", "v1", "test-model", "low", "system", "user", _Answer)


def test_valid_reply_is_parsed_and_cached(tmp_path):
    client = _ScriptedClient(['{"label": "hook", "score": 4}'])
    service = LlmService(client, LlmCacheRepository(tmp_path))
    assert _call(service) == _Answer(label="hook", score=4)
    assert _call(service) == _Answer(label="hook", score=4)
    assert len(client.requests) == 1
    assert [record.is_cache_hit for record in service.call_records] == [False, True]


def test_invalid_reply_gets_one_repair(tmp_path):
    client = _ScriptedClient(['{"label": "hook"}', 'Here you go: {"label": "hook", "score": 2}'])
    service = LlmService(client, LlmCacheRepository(tmp_path))
    assert _call(service) == _Answer(label="hook", score=2)
    assert "previous reply was not valid" in client.requests[1].user_prompt
    assert [record.outcome for record in service.call_records] == ["retried", "repaired"]


def test_still_invalid_or_refused_returns_none(tmp_path):
    service = LlmService(_ScriptedClient(["nope", "still nope"]), LlmCacheRepository(tmp_path / "a"))
    assert _call(service) is None
    refused = LlmService(_ScriptedClient([LlmRefusalError("refused")]), LlmCacheRepository(tmp_path / "b"))
    assert _call(refused) is None
    assert refused.call_records[0].outcome == "failed"


def test_strict_schema_inlines_references_and_closes_objects():
    class _Outer(BaseModel):
        answers: list[_Answer]

    schema = build_strict_json_schema(_Outer)
    item_schema = schema["properties"]["answers"]["items"]
    assert "$defs" not in schema and "$ref" not in str(schema)
    assert item_schema["additionalProperties"] is False
    assert item_schema["required"] == ["label", "score"]
