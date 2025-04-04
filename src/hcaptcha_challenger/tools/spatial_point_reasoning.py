import os
from pathlib import Path
from typing import Union

from google import genai
from google.genai import types
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_fixed

from hcaptcha_challenger.models import SCoTModelType, ImageAreaSelectChallenge
from hcaptcha_challenger.tools.common import extract_first_json_block

THINKING_PROMPT = """
**Thinking step-by-step:**
1. Identify challenge prompt about the Challenge Image
2. Think about what the challenge requires identification goals, and where are they in the picture
3. Based on the plane rectangular coordinate system, reasoning about the absolute position of the "answer object" in the coordinate system

Finally, solve the challenge, locate the object, output the coordinates of the correct answer as json. Follow the following format to return a coordinates wrapped with a json code block:

```json
{
  "challenge_prompt": "Task description",
  "points": [
    {"x": x1, "y": y1}
  ]
}
```
"""


class SpatialPointReasoner:
    def __init__(self, gemini_api_key: str):
        """Initialize the classifier with a Gemini API key."""
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
        grid_divisions: Union[str, Path, os.PathLike],
        auxiliary_information: str | None = "",
        model: SCoTModelType = "gemini-2.5-pro-exp-03-25",
    ) -> ImageAreaSelectChallenge:
        # Initialize Gemini client with API key
        client = genai.Client(api_key=self._api_key)

        # Upload the challenge image file
        files = [
            client.files.upload(file=challenge_screenshot),
            client.files.upload(file=grid_divisions),
        ]

        # Create content with only the image
        parts = [
            types.Part.from_uri(file_uri=files[0].uri, mime_type=files[0].mime_type),
            types.Part.from_uri(file_uri=files[1].uri, mime_type=files[1].mime_type),
        ]
        if auxiliary_information and isinstance(auxiliary_information, str):
            parts.append(types.Part.from_text(text=auxiliary_information))

        contents = [types.Content(role="user", parts=parts)]

        # Change to JSON mode
        if model in ["gemini-2.0-flash-thinking-exp-01-21"]:
            response = client.models.generate_content(
                model=model,
                contents=contents,
                config=types.GenerateContentConfig(
                    temperature=0, system_instruction=THINKING_PROMPT
                ),
            )
            return ImageAreaSelectChallenge(**extract_first_json_block(response.text))

        # Structured output with Constraint encoding
        response = client.models.generate_content(
            model=model,
            contents=contents,
            config=types.GenerateContentConfig(
                temperature=0,
                system_instruction=THINKING_PROMPT,
                response_mime_type="application/json",
                response_schema=ImageAreaSelectChallenge,
            ),
        )

        return ImageAreaSelectChallenge(**response.parsed.model_dump())
