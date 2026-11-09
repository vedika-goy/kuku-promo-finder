"""Any OpenAI-compatible chat endpoint (Gemini, Groq, OpenRouter, OpenAI, or an API provided at interview)."""

import openai

from promo_finder import config
from promo_finder.llm.schemas import LlmCompletion, LlmJsonRequest, LlmUsage
from promo_finder.llm.utils import describe_schema_for_prompt


class OpenAiCompatibleLlmClient:
    def __init__(self, base_url: str, api_key: str) -> None:
        self._client = openai.OpenAI(
            base_url=base_url, api_key=api_key, timeout=config.LLM_TIMEOUT_S, max_retries=config.LLM_MAX_TRANSIENT_RETRIES
        )

    def complete_json(self, request: LlmJsonRequest) -> LlmCompletion:
        # Why not response_format=json_schema: support for strict schemas differs across compatible providers,
        # so the schema goes into the prompt and Pydantic validates the reply.
        response = self._client.chat.completions.create(
            model=request.model,
            max_tokens=request.max_output_tokens,
            messages=[
                {"role": "system", "content": request.system_prompt},
                {"role": "user", "content": f"{request.user_prompt}\n\n{describe_schema_for_prompt(request.json_schema)}"},
            ],
            response_format={"type": "json_object"},
        )
        choice = response.choices[0]
        usage = LlmUsage(
            input_tokens=response.usage.prompt_tokens if response.usage else 0,
            output_tokens=response.usage.completion_tokens if response.usage else 0,
        )
        return LlmCompletion(text=choice.message.content or "", usage=usage, served_by_model=response.model, stop_reason=choice.finish_reason)
