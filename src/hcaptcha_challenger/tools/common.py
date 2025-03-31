import json
import re
from typing import List

from loguru import logger


def extract_json_blocks(text: str) -> List[str]:
    """
    Extract the contents of JSON code blocks surrounded by ```json and ``` from the text.

    Args:
        text: String containing possible JSON code blocks

    Returns:
        Extracted JSON content list, not containing ```json and ``` tags
    """
    try:
        # Use regular expressions to match the content between ```json and ``
        pattern = r"```json\s*([\s\S]*?)```"
        matches = re.findall(pattern, text)

        # If no match is found, record the warning and return to the empty list
        if not matches:
            logger.warning("No JSON code blocks found in the provided text")
            return []

        return matches
    except Exception as e:
        logger.error(f"Error extracting JSON blocks: {str(e)}")
        return []


def extract_first_json_block(text: str) -> dict:
    """
    Extract the first JSON code block contents surrounded by ```json and ``` from the text.

    Args:
        text: String containing possible JSON code blocks

    Returns:
        The first JSON content extracted, if not found, returns None
    """
    if blocks := extract_json_blocks(text):
        return json.loads(blocks[0])
