# -*- coding: utf-8 -*-
"""Shared JSON parsing helpers for chat providers."""
import json
import re


def extract_first_json_block(text: str) -> dict | None:
    """Extract the first ```json fenced block from text."""
    pattern = r"```json\s*([\s\S]*?)```"
    matches = re.findall(pattern, text)
    if matches:
        return json.loads(matches[0])
    return None


def parse_json_response(text: str) -> dict:
    """
    Parse a model text response into a dict, tolerant of common formats.

    Order: raw json.loads -> first ```json fenced block -> ValueError.
    """
    text = (text or "").strip()
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except (json.JSONDecodeError, TypeError):
        pass

    block = extract_first_json_block(text)
    if isinstance(block, dict):
        return block

    raise ValueError(f"Failed to parse JSON response: {text!r}")
