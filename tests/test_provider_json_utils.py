# -*- coding: utf-8 -*-
import pytest

from hcaptcha_challenger.tools.internal.providers._utils import (
    parse_json_response,
    extract_first_json_block,
)


def test_parse_raw_json():
    assert parse_json_response('{"a": 1}') == {"a": 1}


def test_parse_fenced_json():
    text = 'noise\n```json\n{"a": 2}\n```\ntail'
    assert parse_json_response(text) == {"a": 2}


def test_parse_raw_takes_priority_over_fence():
    # A whole-text raw object parses directly.
    assert parse_json_response('{"a": 3}') == {"a": 3}


def test_parse_garbage_raises():
    with pytest.raises(ValueError):
        parse_json_response("not json at all")


def test_extract_first_json_block_none():
    assert extract_first_json_block("no fence here") is None
