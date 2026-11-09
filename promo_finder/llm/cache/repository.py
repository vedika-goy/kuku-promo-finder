"""Disk cache of LLM replies, so reruns are free and repeatable."""

import hashlib
import json
from pathlib import Path

from promo_finder.llm.schemas import LlmCompletion, LlmJsonRequest


class LlmCacheRepository:
    def __init__(self, cache_dir: Path) -> None:
        self._cache_dir = cache_dir
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def get(self, request: LlmJsonRequest, prompt_version: str) -> LlmCompletion | None:
        path = self._build_path(request, prompt_version)
        if not path.exists():
            return None
        return LlmCompletion.model_validate_json(path.read_text(encoding="utf-8"))

    def put(self, request: LlmJsonRequest, prompt_version: str, completion: LlmCompletion) -> None:
        path = self._build_path(request, prompt_version)
        temporary_path = path.with_suffix(".tmp")
        temporary_path.write_text(completion.model_dump_json(), encoding="utf-8")
        temporary_path.replace(path)

    def _build_path(self, request: LlmJsonRequest, prompt_version: str) -> Path:
        key_material = json.dumps(
            [request.prompt_name, prompt_version, request.model, request.effort, request.system_prompt, request.user_prompt, request.json_schema],
            sort_keys=True,
        )
        digest = hashlib.sha256(key_material.encode("utf-8")).hexdigest()[:32]
        return self._cache_dir / f"{request.prompt_name}-{digest}.json"
