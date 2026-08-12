"""Shared helpers for parsing JSON out of LLM responses.

Reasoning models (e.g. gpt-oss on vLLM) intermittently wrap JSON in ```json
fences, prepend/append prose, or emit several objects. `parse_llm_json` strips
fences and, on failure, extracts the FIRST balanced `{...}` object (respecting
string literals) rather than a greedy regex span — so trailing prose or a second
object no longer breaks parsing.
"""

import json
from typing import Optional

# Cap the scan to avoid burning CPU on a huge/garbled response.
_MAX_JSON_SCAN = 200_000


def extract_first_json_object(text: str) -> Optional[str]:
    """Return the first balanced ``{...}`` substring, or None.

    Scans brace depth while respecting double-quoted strings and escapes, so
    braces inside string values don't confuse the balance count.
    """
    if not text:
        return None
    start = text.find('{')
    if start == -1:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        c = text[i]
        if in_str:
            if esc:
                esc = False
            elif c == '\\':
                esc = True
            elif c == '"':
                in_str = False
        else:
            if c == '"':
                in_str = True
            elif c == '{':
                depth += 1
            elif c == '}':
                depth -= 1
                if depth == 0:
                    return text[start:i + 1]
    return None


def _strip_fences(text: str) -> str:
    """Remove a leading ```/```json fence and a trailing ``` fence if present."""
    if text.startswith('```'):
        text = text[3:]
        if text[:4].lower() == 'json':
            text = text[4:]
        stripped = text.rstrip()
        if stripped.endswith('```'):
            text = stripped[:-3]
    return text.strip()


def parse_llm_json(response: Optional[str]) -> dict:
    """Parse a JSON object from an LLM response, tolerating fences/prose.

    Raises json.JSONDecodeError if no JSON object can be recovered (callers rely
    on this specific exception type for fallback handling).
    """
    if not response:
        raise json.JSONDecodeError('empty LLM response', response or '', 0)

    text = response.strip()
    if len(text) > _MAX_JSON_SCAN:
        text = text[:_MAX_JSON_SCAN]

    stripped = _strip_fences(text)

    try:
        return json.loads(stripped)
    except (json.JSONDecodeError, TypeError):
        candidate = extract_first_json_object(stripped)
        if candidate is not None:
            return json.loads(candidate)
        # Try the un-stripped text too (fence stripping may have mangled a rare case)
        candidate = extract_first_json_object(text)
        if candidate is not None:
            return json.loads(candidate)
        raise
