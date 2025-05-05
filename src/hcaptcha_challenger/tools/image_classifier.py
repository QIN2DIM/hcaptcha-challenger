import os
from pathlib import Path
from typing import Union

from google import genai
from google.genai import types
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_fixed

from hcaptcha_challenger.models import SCoTModelType, ImageBinaryChallenge
from hcaptcha_challenger.tools.common import extract_first_json_block
from hcaptcha_challenger.tools.reasoner import _Reasoner

SYSTEM_INSTRUCTION = """
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


class ImageClassifier(_Reasoner):
    """
    A classifier that uses Google's Gemini AI models to analyze and solve image-based challenges.

    This class provides functionality to process screenshots of binary image challenges
    (typically grid-based selection challenges) and determines the correct answer coordinates.
    """

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_fixed(3),
        before_sleep=lambda retry_state: logger.warning(
            f"Retry request ({retry_state.attempt_number}/2) - Wait 3 seconds - Exception: {retry_state.outcome.exception()}"
        ),
    )
    async def invoke_async(
        self,
        challenge_screenshot: Union[str, Path, os.PathLike],
        model: SCoTModelType = "gemini-2.5-pro-exp-03-25",
        *,
        constraint_response_schema: bool = False,
        **kwargs,
    ) -> ImageBinaryChallenge:
        """
        Process an image challenge and return the solution coordinates.

        Args:
            constraint_response_schema:
            challenge_screenshot: The image file containing the challenge to solve
            model: The Gemini model to use for processing. Must support both visual
               capabilities and Spatial chain-of-thought (S-COT) reasoning.

        Returns:
            ImageBinaryChallenge: Object containing the solution coordinates
        """
        enable_response_schema = kwargs.get("enable_response_schema")
        if enable_response_schema is not None:
            constraint_response_schema = enable_response_schema

        # Initialize Gemini client with API_KEY
        client = genai.Client(api_key=self._api_key)

        # Upload the challenge image file
        files = [await client.aio.files.upload(file=challenge_screenshot)]

        parts = [types.Part.from_uri(file_uri=files[0].uri, mime_type=files[0].mime_type)]
        contents = [types.Content(role="user", parts=parts)]

        # Change to JSON mode
        if not constraint_response_schema or model in ["gemini-2.0-flash-thinking-exp-01-21"]:
            self._response = await client.aio.models.generate_content(
                model=model,
                contents=contents,
                config=types.GenerateContentConfig(
                    temperature=0, system_instruction=SYSTEM_INSTRUCTION
                ),
            )
            return ImageBinaryChallenge(**extract_first_json_block(self._response.text))

        # Handle models that support JSON response schema
        parts.append(types.Part.from_text(text=USER_PROMPT.strip()))

        # Structured output with Constraint encoding
        self._response = await client.aio.models.generate_content(
            model=model,
            contents=contents,
            config=types.GenerateContentConfig(
                temperature=0,
                response_mime_type="application/json",
                response_schema=ImageBinaryChallenge,
            ),
        )
        if _result := self._response.parsed:
            return ImageBinaryChallenge(**self._response.parsed.model_dump())
        return ImageBinaryChallenge(**extract_first_json_block(self._response.text))