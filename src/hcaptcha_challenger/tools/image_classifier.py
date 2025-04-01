import os
from pathlib import Path
from typing import Union

from google import genai
from google.genai import types

from hcaptcha_challenger.models import VCOTModelType, ImageBinaryChallenge
from hcaptcha_challenger.tools.common import extract_first_json_block

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

USER_PROMPT = """
Solve the challenge, use [0,0] ~ [2,2] to locate 9grid, output the coordinates of the correct answer as json.
"""


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
        challenge_screenshot: Union[str, Path, os.PathLike],
        model: VCOTModelType = "gemini-2.0-flash-thinking-exp-01-21",
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

        # Change to JSON mode
        if model in ["gemini-2.0-flash-thinking-exp-01-21"]:
            contents = [
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_uri(file_uri=files[0].uri, mime_type=files[0].mime_type)
                    ],
                )
            ]
            response = client.models.generate_content(
                model=model,
                contents=contents,
                config=types.GenerateContentConfig(
                    temperature=0, system_instruction=THINKING_PROMPT
                ),
            )
            return ImageBinaryChallenge(**extract_first_json_block(response.text))

        # Handle models that support JSON response schema
        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_uri(file_uri=files[0].uri, mime_type=files[0].mime_type),
                    types.Part.from_text(text=USER_PROMPT.strip()),
                ],
            )
        ]

        # Structured output with Constraint encoding
        response = client.models.generate_content(
            model=model,
            contents=contents,
            config=types.GenerateContentConfig(
                temperature=0,
                response_mime_type="application/json",
                response_schema=ImageBinaryChallenge,
            ),
        )

        return ImageBinaryChallenge(**response.parsed.model_dump())
