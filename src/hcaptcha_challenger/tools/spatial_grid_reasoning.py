from typing import Union
from pathlib import Path
import os

from google import genai
from google.genai import types

from hcaptcha_challenger.models import VCOTModelType, ImageBinaryChallenge
from hcaptcha_challenger.tools.common import extract_first_json_block

THINKING_PROMPT = """
Identify the correct area to drag the element into for optimal proximity to its matching puzzle piece's gap.

Solve the challenge, use [0,0] ~ [3,3] to locate 9grid, output the coordinates of the correct answer as json.

Follow the following format to return a coordinates wrapped with a json code block:
```json
{
  "challenge_prompt": "Please click, hold, and drag the element on the right to the shape that is most similar",
  "coordinates": [{"box_2d": [0,0]}]
}
```
"""

USER_PROMPT = """
Identify the correct area to drag the element into for optimal proximity to its matching puzzle piece's gap.

Solve the challenge, use [0,0] ~ [3,3] to locate 9grid, output the coordinates of the correct answer as json.
"""


class SpatialGridReasoner:
    def __init__(self, gemini_api_key: str):
        """Initialize the classifier with a Gemini API key."""
        self._api_key = gemini_api_key

    def invoke(
        self,
        challenge_screenshot: Union[str, Path, os.PathLike],
        grid_divisions: Union[str, Path, os.PathLike],
        model: VCOTModelType = "gemini-2.0-flash-thinking-exp-01-21",
    ):
        # Initialize Gemini client with API key
        client = genai.Client(api_key=self._api_key)

        # Upload the challenge image file
        files = [
            client.files.upload(file=challenge_screenshot),
            client.files.upload(file=grid_divisions),
        ]

        # Handle models that don't support JSON response schema
        if model in ["gemini-2.0-flash-thinking-exp-01-21"]:
            # Create content with only the image
            contents = [
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_uri(file_uri=files[0].uri, mime_type=files[0].mime_type),
                        types.Part.from_uri(file_uri=files[1].uri, mime_type=files[1].mime_type),
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
            print(response.text)
            # Extract and parse JSON from text response
            return ImageBinaryChallenge(**extract_first_json_block(response.text))

        # Handle models that support JSON response schema
        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_uri(file_uri=files[0].uri, mime_type=files[0].mime_type),
                    types.Part.from_uri(file_uri=files[1].uri, mime_type=files[1].mime_type),
                    types.Part.from_text(text=USER_PROMPT.strip()),
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
