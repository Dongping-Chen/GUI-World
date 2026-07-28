from __future__ import annotations

import json
import re
from typing import Any


class ResponseParseError(ValueError):
    """Raised when a provider response cannot be normalized."""


_FENCED_JSON = re.compile(
    r"```(?:json)?\s*(?P<content>.*?)\s*```",
    flags=re.IGNORECASE | re.DOTALL,
)


def extract_json_object(text: str) -> dict[str, Any]:
    """Return the first valid JSON object embedded in *text*."""

    decoder = json.JSONDecoder()
    for index, character in enumerate(text):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text, index)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ResponseParseError("Response does not contain a valid JSON object.")


def normalize_answer(value: dict[str, Any] | str) -> dict[str, str]:
    """Normalize a provider response to supported string fields."""

    if isinstance(value, dict):
        response = value
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            raise ResponseParseError("Response is empty.")
        if text.startswith("```"):
            match = _FENCED_JSON.fullmatch(text)
            if match is None:
                raise ResponseParseError("Response contains malformed fenced JSON.")
            try:
                response = json.loads(match.group("content"))
            except json.JSONDecodeError as error:
                raise ResponseParseError(
                    "Response contains malformed fenced JSON."
                ) from error
            if not isinstance(response, dict):
                raise ResponseParseError("JSON response must be an object.")
        else:
            try:
                decoded = json.loads(text)
            except json.JSONDecodeError:
                try:
                    response = extract_json_object(text)
                except ResponseParseError:
                    return {"Answer": text}
            else:
                if not isinstance(decoded, dict):
                    raise ResponseParseError("JSON response must be an object.")
                response = decoded
    else:
        raise ResponseParseError("Response must be a dictionary or string.")

    answer_key = next(
        (
            key
            for key in response
            if isinstance(key, str) and key.casefold() == "answer"
        ),
        None,
    )
    if answer_key is None:
        raise ResponseParseError("JSON response is missing an Answer field.")

    answer = response[answer_key]
    if not isinstance(answer, str):
        raise ResponseParseError("Answer must be a string.")

    normalized = {
        key: response[key]
        for key in ("Description", "Analysis")
        if isinstance(response.get(key), str)
    }
    normalized["Answer"] = answer
    return normalized
