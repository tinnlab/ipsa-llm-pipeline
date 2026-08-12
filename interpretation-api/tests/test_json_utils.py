"""
Tests for the shared robust LLM-JSON parser (json_utils).

Covers the balanced-brace extraction that replaced a greedy `\\{.*\\}` regex:
trailing prose, multiple objects, and braces inside string values.
"""

import json
import pytest

from src.pipeline.json_utils import parse_llm_json, extract_first_json_object


def test_plain_and_fenced():
    assert parse_llm_json('{"a": 1}') == {'a': 1}
    assert parse_llm_json('```json\n{"a": 1}\n```') == {'a': 1}
    assert parse_llm_json('```\n{"a": 1}\n```') == {'a': 1}


def test_prose_prefix_and_suffix():
    assert parse_llm_json('Here you go:\n{"a": 1}\nThanks!') == {'a': 1}


def test_trailing_prose_with_braces_recovers_first_object():
    # Greedy regex would span to the last brace and fail; balanced scan gets the first object.
    assert parse_llm_json('{"a": 1} note: use {curly}') == {'a': 1}


def test_multiple_objects_returns_first():
    assert parse_llm_json('prose {"a": 1} and {"b": 2}') == {'a': 1}


def test_braces_inside_string_value():
    assert parse_llm_json('{"code": "if (x) { y }"}') == {'code': 'if (x) { y }'}


def test_fenced_with_backticks_in_string():
    out = parse_llm_json('```json\n{"code": "a ```b``` c"}\n```')
    assert out['code'] == 'a ```b``` c'


def test_empty_and_none_raise():
    with pytest.raises(json.JSONDecodeError):
        parse_llm_json('')
    with pytest.raises(json.JSONDecodeError):
        parse_llm_json(None)


def test_no_json_raises():
    with pytest.raises(json.JSONDecodeError):
        parse_llm_json('sorry, I cannot help with that')


def test_extract_first_json_object_none_when_absent():
    assert extract_first_json_object('no braces here') is None
    assert extract_first_json_object('') is None


def test_unbalanced_open_brace_does_not_hang():
    # Many open braces, no close — must terminate and raise, not hang.
    with pytest.raises(json.JSONDecodeError):
        parse_llm_json('{' * 5000)


def test_oversized_input_is_capped():
    """Input beyond the scan cap is truncated; a valid object within the cap parses,
    and huge junk past the cap doesn't blow up."""
    from src.pipeline.json_utils import _MAX_JSON_SCAN
    # valid object, then a massive tail of junk well beyond the cap
    payload = '{"a": 1}' + (' x' * _MAX_JSON_SCAN)
    assert parse_llm_json(payload) == {'a': 1}
    # object that only completes AFTER the cap -> not recoverable
    big = '{"pad": "' + ('y' * (_MAX_JSON_SCAN + 100)) + '"}'
    with pytest.raises(json.JSONDecodeError):
        parse_llm_json(big)
