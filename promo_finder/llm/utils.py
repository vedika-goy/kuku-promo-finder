"""Turn a Pydantic model into the JSON schema subset that strict structured-output modes accept."""

import copy
import json
import re
from typing import Any

from pydantic import BaseModel

# Keywords strict modes reject or ignore; Pydantic still enforces them locally after the call.
_UNSUPPORTED_KEYWORDS = {"default", "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "minLength", "maxLength", "minItems", "maxItems", "pattern", "format", "title"}
_JSON_OBJECT = re.compile(r"\{.*\}", re.S)


def build_strict_json_schema(model_class: type[BaseModel]) -> dict[str, Any]:
    raw_schema = model_class.model_json_schema()
    definitions = raw_schema.pop("$defs", {})
    return _strictify(_inline_references(raw_schema, definitions))


def extract_json_text(text: str) -> str:
    """The JSON object inside a reply that may carry code fences or a sentence around it."""
    stripped_text = text.strip()
    if stripped_text.startswith("{"):
        return stripped_text
    match = _JSON_OBJECT.search(stripped_text)
    return match.group(0) if match else stripped_text


def _inline_references(node: Any, definitions: dict[str, Any]) -> Any:
    if isinstance(node, dict):
        if "$ref" in node:
            definition_name = node["$ref"].split("/")[-1]
            return _inline_references(copy.deepcopy(definitions[definition_name]), definitions)
        return {key: _inline_references(value, definitions) for key, value in node.items()}
    if isinstance(node, list):
        return [_inline_references(item, definitions) for item in node]
    return node


def _strictify(node: Any) -> Any:
    if isinstance(node, list):
        return [_strictify(item) for item in node]
    if not isinstance(node, dict):
        return node
    cleaned = {key: _strictify(value) for key, value in node.items() if key not in _UNSUPPORTED_KEYWORDS}
    if cleaned.get("type") == "object" and "properties" in cleaned:
        cleaned["additionalProperties"] = False
        cleaned["required"] = list(cleaned["properties"].keys())
    return cleaned


def describe_schema_for_prompt(json_schema: dict[str, Any]) -> str:
    """Used by providers without a schema mode: the schema goes into the prompt instead."""
    return "Reply with one JSON object only, matching this JSON schema:\n" + json.dumps(json_schema, indent=1)
