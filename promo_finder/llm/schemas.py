from typing import Any

from pydantic import BaseModel


class LlmJsonRequest(BaseModel):
    """One JSON-producing call. `json_schema` constrains the reply where the provider supports it."""

    prompt_name: str
    model: str
    effort: str  # low | medium | high (ignored by providers without an effort setting)
    system_prompt: str
    user_prompt: str
    json_schema: dict[str, Any]
    max_output_tokens: int


class LlmUsage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0


class LlmCompletion(BaseModel):
    text: str
    usage: LlmUsage
    served_by_model: str
    stop_reason: str | None = None


class LlmCallRecord(BaseModel):
    """One line of the run log's LLM section."""

    prompt_name: str
    model: str
    is_cache_hit: bool
    input_tokens: int
    output_tokens: int
    outcome: str  # ok | retried (first reply invalid) | repaired | invalid | failed
    detail: str = ""
