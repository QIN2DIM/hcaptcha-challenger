import os
from enum import Enum
from pathlib import Path
from typing import Union

from google import genai
from google.genai import types
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_fixed

from hcaptcha_challenger.models import FastShotModelType

CHALLENGE_CLASSIFIER_INSTRUCTIONS = """
# Instructions

Your task is to classify challenge questions into one of four types:

1. `image_label_single_select`: Requires clicking on a SINGLE specific area/object of an image based on a prompt
2. `image_label_multi_select`: Requires clicking on MULTIPLE areas/objects of an image based on a prompt
3. `image_drag_single`: Requires dragging a SINGLE puzzle piece/element to a specific location on an image
4. `image_drag_multi`: Requires dragging MULTIPLE puzzle pieces/elements to specific locations on an image

## Rules

- Output ONLY one of the four classification types listed above
- Do not provide any explanations, reasoning, or additional text
- For clicking/selecting tasks:
  - If the question implies selecting ONE item/area, output `image_label_single_select`
  - If the question implies selecting MULTIPLE items/areas, output `image_label_multi_select`
  - IF the question implies 9grid selection, output `image_label_multi_select`
- For dragging tasks:
  - If the question implies dragging ONE item/element, output `image_drag_single`
  - If the question implies dragging MULTIPLE items/elements, output `image_drag_multi`

## Examples

Input: "Please click on the object that is different from the others"
Output: `image_label_single_select`

Input: "Please click on the two elements that are identical"
Output: `image_label_multi_select`

Input: "Please drag the puzzle piece to complete the image"
Output: `image_drag_single`

Input: "Arrange all the shapes by dragging them to their matching outlines"
Output: `image_drag_multi`
"""

USER_PROMPT = """
Your task is to classify challenge questions into one of four types: 
    - image_label_single_select (clicking ONE specific area/object)
    - image_label_multi_select (clicking MULTIPLE areas/objects)
    - image_drag_single (dragging ONE element/piece)
    - image_drag_multi (dragging MULTIPLE elements/pieces)
"""


class ChallengeTypeEnum(str, Enum):
    IMAGE_LABEL_SINGLE_SELECT = "image_label_single_select"
    IMAGE_LABEL_MULTI_SELECT = "image_label_multi_select"
    IMAGE_DRAG_SINGLE = "image_drag_single"
    IMAGE_DRAG_MULTI = "image_drag_multi"


class ChallengeClassifier:
    def __init__(self, gemini_api_key: str):
        self._api_key = gemini_api_key

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_fixed(3),
        before_sleep=lambda retry_state: logger.warning(
            f"Retry request ({retry_state.attempt_number}/2) - Wait 3 seconds - Exception: {retry_state.outcome.exception()}"
        ),
    )
    def invoke(
        self,
        challenge_screenshot: Union[str, Path, os.PathLike],
        model: FastShotModelType = "gemini-2.0-flash",
    ) -> ChallengeTypeEnum:
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
                    temperature=0, system_instruction=CHALLENGE_CLASSIFIER_INSTRUCTIONS
                ),
            )
            # Extract and parse JSON from text response
            return ChallengeTypeEnum(response.text)

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
        # Generate structured JSON response
        response = client.models.generate_content(
            model=model,
            contents=contents,
            config=types.GenerateContentConfig(
                temperature=0, response_mime_type="text/x.enum", response_schema=ChallengeTypeEnum
            ),
        )

        # Return parsed response as ImageBinaryChallenge object
        return ChallengeTypeEnum(response.text)
