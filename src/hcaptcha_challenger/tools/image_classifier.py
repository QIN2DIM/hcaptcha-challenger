import io
import json
import os
import re
from pathlib import Path
from typing import List
from typing import Union, Literal

from google import genai
from google.genai import types
from loguru import logger
from pydantic import BaseModel, Field

THINKING_PROMPT = """
Solve the challenge, use [0,0] ~ [2,2] to locate 9grid, output the coordinates of the correct answer as json.

Follow the following format to return a coordinates wrapped with a json code block:
```json
{
  "challenge_prompt": "please click on the largest animal",
  "coordinates": [
    {"box_2d": [0,0]},
    {"box_2d": [1,2]},
    {"box_2d": [2,1]}
  ]
}
```
"""


class BoundingBoxCoordinate(BaseModel):
    box_2d: List[int] = Field(
        description="It can only be in planar coordinate format, e.g. [0,2] for the 3rd element in the first row"
    )


class ImageBinaryChallenge(BaseModel):
    challenge_prompt: str
    coordinates: List[BoundingBoxCoordinate]

    def convert_box_to_boolean_matrix(self) -> List[bool]:
        """
        Converts the coordinate list to a one-dimensional Boolean matrix.

        Convert coordinates in a 3x3 matrix to a one-dimensional boolean list where:
        - [0,0] Corresponding index 0
        - [0,1] Corresponding index 1
        - ...
        - [2,2] Corresponding index 8

        Returns:
            List[bool]: Boolean list with length 9, coordinate position is True, other positions are False
        """
        # Initialize a boolean list of length 9, all False
        result = [False] * 9

        for coord in self.coordinates:
            row, col = coord.box_2d

            if 0 <= row < 3 and 0 <= col < 3:
                index = row * 3 + col
                result[index] = True

        return result

    @property
    def log_message(self) -> str:
        _coordinates = [i.box_2d for i in self.coordinates]
        bundle = {"Challenge Prompt": self.challenge_prompt, "Coordinates": str(_coordinates)}
        return json.dumps(bundle, indent=2, ensure_ascii=False)


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


class ImageClassifier:
    """
    A classifier that uses Google's Gemini AI models to analyze and solve image-based challenges.

    This class provides functionality to process screenshots of binary image challenges
    (typically grid-based selection challenges) and determine the correct answer coordinates.
    """

    def __init__(self, gemini_api_key: str):
        """Initialize the classifier with a Gemini API key."""
        self._api_key = gemini_api_key

    def invoke(
            self,
            challenge_screenshot: Union[str, Path, os.PathLike, io.IOBase],
            model: Literal[
                "gemini-2.5-pro-exp-03-25", "gemini-2.0-flash-thinking-exp-01-21"
            ] = "gemini-2.0-flash-thinking-exp-01-21",
    ) -> ImageBinaryChallenge:
        """
        Process an image challenge and return the solution coordinates.

        The method handles two different Gemini model approaches:
        1. For "gemini-2.0-flash-thinking-exp-01-21": Uses a thinking prompt and extracts JSON from text response
        2. For other models: Uses structured JSON response schema directly

        Args:
            challenge_screenshot: The image file containing the challenge to solve
            model: The Gemini model to use for processing. Must support both visual
               capabilities and chain-of-thought (COT) reasoning. The default
               "gemini-2.0-flash-thinking-exp-01-21" is optimized for spatial
               reasoning with visual inputs, while "gemini-2.5-pro-exp-03-25"
               offers enhanced visual understanding with structured outputs.

        Returns:
            ImageBinaryChallenge: Object containing the solution coordinates
        """
        # Initialize Gemini client with API key
        client = genai.Client(api_key=self._api_key)

        # Upload the challenge image file
        files = [client.files.upload(file=challenge_screenshot)]

        # Handle models that don't support JSON response schema
        if model in ["gemini-2.0-flash-thinking-exp-01-21"]:
            # Create content with only the image
            contents = [
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_uri(file_uri=files[0].uri, mime_type=files[0].mime_type)
                    ],
                )
            ]
            # Generate response using thinking prompt
            response = client.models.generate_content(
                model=model,
                contents=contents,
                config=types.GenerateContentConfig(
                    temperature=0, system_instruction=THINKING_PROMPT
                ),
            )
            # Extract and parse JSON from text response
            return ImageBinaryChallenge(**extract_first_json_block(response.text))

        # Handle models that support JSON response schema
        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_uri(file_uri=files[0].uri, mime_type=files[0].mime_type),
                    types.Part.from_text(
                        text="""Solve the challenge, use [0,0] ~ [2,2] to locate 9grid, output the coordinates of the correct answer as json."""
                    ),
                ],
            )
        ]
        # Generate structured JSON response
        response = client.models.generate_content(
            model=model,
            contents=contents,
            config=types.GenerateContentConfig(
                temperature=0,
                response_mime_type="application/json",
                response_schema=ImageBinaryChallenge,
            ),
        )

        # Return parsed response as ImageBinaryChallenge object
        return ImageBinaryChallenge(**response.parsed.model_dump())
