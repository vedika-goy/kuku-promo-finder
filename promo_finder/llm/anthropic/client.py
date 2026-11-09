"""Claude through the official Anthropic SDK."""

import logging

import anthropic

from promo_finder import config
from promo_finder.llm.schemas import LlmCompletion, LlmJsonRequest, LlmUsage
from promo_finder.llm.utils import describe_schema_for_prompt

logger = logging.getLogger(__name__)

# Server-side refusal fallback ("default" picks Anthropic's recommended substitute per refusal category).
# Only these models accept it; others get a plain request.
_FALLBACK_BETA = "server-side-fallback-2026-07-01"
_FALLBACK_ENABLED_MODELS = {"claude-opus-5-5", "claude-sonnet-5-5", "claude-opus-5", "claude-fable-5-1"}


class LlmRefusalError(RuntimeError):
    pass


class LlmTruncatedError(RuntimeError):
    pass


class AnthropicLlmClient:
    def __init__(self) -> None:
        # The SDK retries connection errors, 408, 409, 429 and 5xx with exponential backoff.
        self._client = anthropic.Anthropic(timeout=config.LLM_TIMEOUT_S, max_retries=config.LLM_MAX_TRANSIENT_RETRIES)

    def complete_json(self, request: LlmJsonRequest) -> LlmCompletion:
        try:
            return self._stream(self._build_params(request, is_schema_enforced=True))
        except anthropic.BadRequestError as error:
            # Why not fail here: a model or account without structured outputs rejects output_config.format;
            # the schema then goes into the prompt and Pydantic validates the reply.
            logger.warning("%s: structured output rejected (%s); retrying with the schema in the prompt", request.prompt_name, error.message)
            return self._stream(self._build_params(request, is_schema_enforced=False))

    def _build_params(self, request: LlmJsonRequest, is_schema_enforced: bool) -> dict:
        output_config: dict = {"effort": request.effort}
        user_prompt = request.user_prompt
        if is_schema_enforced:
            output_config["format"] = {"type": "json_schema", "schema": request.json_schema}
        else:
            user_prompt = f"{request.user_prompt}\n\n{describe_schema_for_prompt(request.json_schema)}"
        params = {
            "model": request.model,
            "max_tokens": request.max_output_tokens,
            "system": request.system_prompt,
            "messages": [{"role": "user", "content": user_prompt}],
            "output_config": output_config,
        }
        if request.model in _FALLBACK_ENABLED_MODELS:
            params["betas"] = [_FALLBACK_BETA]
            params["fallbacks"] = "default"
        return params

    def _stream(self, params: dict) -> LlmCompletion:
        # Streaming keeps long thinking-plus-output replies clear of HTTP timeouts.
        with self._client.beta.messages.stream(**params) as stream:
            message = stream.get_final_message()
        if message.stop_reason == "refusal":
            category = message.stop_details.category if message.stop_details else None
            raise LlmRefusalError(f"model refused (category={category})")
        if message.stop_reason == "max_tokens":
            raise LlmTruncatedError(f"reply hit max_tokens={params['max_tokens']}")
        text = next((block.text for block in message.content if block.type == "text"), "")
        usage = LlmUsage(input_tokens=message.usage.input_tokens, output_tokens=message.usage.output_tokens)
        return LlmCompletion(text=text, usage=usage, served_by_model=message.model, stop_reason=message.stop_reason)
