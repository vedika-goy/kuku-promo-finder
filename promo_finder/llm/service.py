"""One way to call an LLM for JSON: cache, call, validate against a Pydantic model, repair once, log."""

import logging
import threading
from typing import Protocol, TypeVar

import anthropic
import openai
from pydantic import BaseModel, ValidationError

from promo_finder.config import MODEL_PRICES_USD_PER_MILLION_TOKENS, ModelSettings
from promo_finder.llm.anthropic.client import AnthropicLlmClient, LlmRefusalError, LlmTruncatedError
from promo_finder.llm.cache.repository import LlmCacheRepository
from promo_finder.llm.offline.client import OfflineLlmClient
from promo_finder.llm.openai_compatible.client import OpenAiCompatibleLlmClient
from promo_finder.llm.schemas import LlmCallRecord, LlmCompletion, LlmJsonRequest
from promo_finder.llm.utils import build_strict_json_schema, extract_json_text

logger = logging.getLogger(__name__)

OutputModel = TypeVar("OutputModel", bound=BaseModel)

# Errors that affect every call (bad key, unknown model): stop the run instead of skipping calls one by one.
_FATAL_ERRORS = (anthropic.AuthenticationError, anthropic.PermissionDeniedError, anthropic.NotFoundError, openai.AuthenticationError, openai.NotFoundError)
_CALL_ERRORS = (anthropic.APIError, openai.APIError, LlmRefusalError, LlmTruncatedError)


class LlmClient(Protocol):
    def complete_json(self, request: LlmJsonRequest) -> LlmCompletion: ...


def build_llm_client(settings: ModelSettings) -> LlmClient:
    if settings.provider == "anthropic":
        return AnthropicLlmClient()
    if settings.provider == "openai_compatible":
        if not settings.openai_compatible_base_url or not settings.openai_compatible_api_key:
            raise ValueError("LLM_PROVIDER=openai_compatible needs OPENAI_COMPATIBLE_BASE_URL and OPENAI_COMPATIBLE_API_KEY")
        return OpenAiCompatibleLlmClient(settings.openai_compatible_base_url, settings.openai_compatible_api_key)
    if settings.provider == "offline":
        return OfflineLlmClient()
    raise ValueError(f"Unknown LLM_PROVIDER '{settings.provider}' (use anthropic, openai_compatible or offline)")


class LlmService:
    def __init__(self, client: LlmClient, cache_repository: LlmCacheRepository) -> None:
        self._client = client
        self._cache_repository = cache_repository
        self._records: list[LlmCallRecord] = []
        self._records_lock = threading.Lock()

    def call_json(
        self,
        prompt_name: str,
        prompt_version: str,
        model: str,
        effort: str,
        system_prompt: str,
        user_prompt: str,
        output_model: type[OutputModel],
        max_output_tokens: int = 16000,
    ) -> OutputModel | None:
        """The validated reply, or None after logging why (the caller decides the fallback)."""
        request = LlmJsonRequest(
            prompt_name=prompt_name, model=model, effort=effort, system_prompt=system_prompt, user_prompt=user_prompt,
            json_schema=build_strict_json_schema(output_model), max_output_tokens=max_output_tokens,
        )
        cached_completion = self._cache_repository.get(request, prompt_version)
        if cached_completion:
            parsed, _ = _parse_reply(cached_completion.text, output_model)
            if parsed is not None:
                self._record(request, cached_completion, is_cache_hit=True, outcome="ok")
                return parsed
        completion = self._complete_or_none(request)
        if completion is None:
            return None
        parsed, error_text = _parse_reply(completion.text, output_model)
        if parsed is None:
            self._record(request, completion, is_cache_hit=False, outcome="retried", detail=error_text[:300])
            parsed, completion = self._repair(request, error_text, output_model)
            if parsed is None:
                return None
        else:
            self._record(request, completion, is_cache_hit=False, outcome="ok")
        self._cache_repository.put(request, prompt_version, completion)
        return parsed

    @property
    def call_records(self) -> list[LlmCallRecord]:
        with self._records_lock:
            return list(self._records)

    def summarize_usage(self) -> dict:
        summary: dict[str, dict] = {}
        for record in self.call_records:
            entry = summary.setdefault(record.prompt_name, {"calls": 0, "cache_hits": 0, "input_tokens": 0, "output_tokens": 0, "failures": 0})
            entry["calls"] += 1
            entry["cache_hits"] += int(record.is_cache_hit)
            entry["failures"] += int(record.outcome in ("invalid", "failed"))
            if not record.is_cache_hit:
                entry["input_tokens"] += record.input_tokens
                entry["output_tokens"] += record.output_tokens
        live_records = [record for record in self.call_records if not record.is_cache_hit]
        estimated_cost_usd = sum(_estimate_cost_usd(record) for record in live_records)
        return {"by_prompt": summary, "live_calls": len(live_records), "estimated_cost_usd": round(estimated_cost_usd, 4)}

    def _complete_or_none(self, request: LlmJsonRequest) -> LlmCompletion | None:
        try:
            completion = self._client.complete_json(request)
        except _FATAL_ERRORS:
            raise
        except _CALL_ERRORS as error:
            logger.error("LLM call %s on %s failed: %s", request.prompt_name, request.model, error)
            self._record(request, None, is_cache_hit=False, outcome="failed", detail=str(error)[:300])
            return None
        return completion

    def _repair(self, request: LlmJsonRequest, error_text: str, output_model: type[OutputModel]) -> tuple[OutputModel | None, LlmCompletion | None]:
        logger.warning("LLM reply for %s failed validation; asking once more: %s", request.prompt_name, error_text[:300])
        repair_request = request.model_copy(update={
            "user_prompt": f"{request.user_prompt}\n\nYour previous reply was not valid: {error_text[:1500]}\nReply again with one valid JSON object only.",
        })
        completion = self._complete_or_none(repair_request)
        if completion is None:
            return None, None
        parsed, second_error_text = _parse_reply(completion.text, output_model)
        if parsed is None:
            logger.error("LLM reply for %s still invalid after one repair: %s", request.prompt_name, second_error_text[:300])
            self._record(request, completion, is_cache_hit=False, outcome="invalid", detail=second_error_text[:300])
            return None, None
        self._record(request, completion, is_cache_hit=False, outcome="repaired")
        return parsed, completion

    def _record(self, request: LlmJsonRequest, completion: LlmCompletion | None, is_cache_hit: bool, outcome: str, detail: str = "") -> None:
        record = LlmCallRecord(
            prompt_name=request.prompt_name,
            model=completion.served_by_model if completion else request.model,
            is_cache_hit=is_cache_hit,
            input_tokens=completion.usage.input_tokens if completion else 0,
            output_tokens=completion.usage.output_tokens if completion else 0,
            outcome=outcome,
            detail=detail,
        )
        with self._records_lock:
            self._records.append(record)


def _parse_reply(text: str, output_model: type[OutputModel]) -> tuple[OutputModel | None, str]:
    try:
        return output_model.model_validate_json(extract_json_text(text)), ""
    except ValidationError as error:
        return None, str(error)


def _estimate_cost_usd(record: LlmCallRecord) -> float:
    input_price, output_price = MODEL_PRICES_USD_PER_MILLION_TOKENS.get(record.model, (0.0, 0.0))
    return (record.input_tokens * input_price + record.output_tokens * output_price) / 1_000_000
