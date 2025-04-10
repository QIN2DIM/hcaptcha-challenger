import json
import os
from pathlib import Path
from typing import Union

from google import genai
from google.genai import types
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_fixed

from hcaptcha_challenger.models import SCoTModelType, ImageBboxChallenge
from hcaptcha_challenger.tools.common import extract_first_json_block

SYSTEM_INSTRUCTIONS = """
<Instruction>
Analyze the input image (which includes a visible coordinate grid) and the accompanying challenge prompt text.
First, interpret the challenge prompt to understand the task or identification required, focusing on the main interactive challenge canvas.
Second, identify the precise target area on the main challenge canvas that represents the answer or the location most relevant to fulfilling the challenge. This target should be enclosed within its minimal possible bounding box.
Finally, output the original challenge prompt and the absolute pixel bounding box coordinates (as integers, based on the image's coordinate grid) for this minimal target area.
</Instruction>

<Output>
{
    "challenge_prompt": "{task_instructions}",
    "bounding_box": {
      "top_left_x": 148,     
      "top_left_y": 260,     
      "bottom_right_x": 235, 
      "bottom_right_y": 345  
    }
}
</Output>
"""


class SpatialBboxReasoner:
    def __init__(self, gemini_api_key: str):
        """Initialize the classifier with a Gemini API key."""
        self._api_key = gemini_api_key

    # @retry(
    #     stop=stop_after_attempt(3),
    #     wait=wait_fixed(3),
    #     before_sleep=lambda retry_state: logger.warning(
    #         f"Retry request ({retry_state.attempt_number}/2) - Wait 3 seconds - Exception: {retry_state.outcome.exception()}"
    #     ),
    # )
    def invoke(
        self,
        challenge_screenshot: Union[str, Path, os.PathLike],
        grid_divisions: Union[str, Path, os.PathLike],
        auxiliary_information: str | None = "",
        model: SCoTModelType = "gemini-2.5-pro-exp-03-25",
        *,
        enable_response_schema: bool = False,
    ) -> ImageBboxChallenge:
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
        if not enable_response_schema or model in ["gemini-2.0-flash-thinking-exp-01-21"]:
            response = client.models.generate_content(
                model=model,
                contents=contents,
                config=types.GenerateContentConfig(
                    temperature=0, system_instruction=SYSTEM_INSTRUCTIONS
                ),
            )

            return ImageBboxChallenge(**extract_first_json_block(response.text))

        # Structured output with Constraint encoding
        response = client.models.generate_content(
            model=model,
            contents=contents,
            config=types.GenerateContentConfig(
                temperature=0,
                system_instruction=SYSTEM_INSTRUCTIONS,
                response_mime_type="application/json",
                response_schema=ImageBboxChallenge,
            ),
        )
        print(json.dumps(response.model_dump(mode="json"), indent=2, ensure_ascii=False))
        if _result := response.parsed:
            return ImageBboxChallenge(**response.parsed.model_dump())
        return ImageBboxChallenge(**extract_first_json_block(response.text))
