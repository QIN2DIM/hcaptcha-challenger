import asyncio
import os
from pathlib import Path
from typing import Union

from google import genai
from google.genai import types
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_fixed

from hcaptcha_challenger.models import SCoTModelType, ImageBboxChallenge
from hcaptcha_challenger.tools.common import extract_first_json_block
from hcaptcha_challenger.tools.reasoner import _Reasoner

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


class SpatialBboxReasoner(_Reasoner[SCoTModelType]):

    def __init__(
        self,
        gemini_api_key: str,
        model: SCoTModelType = "gemini-2.5-pro-exp-03-25",
        constraint_response_schema: bool = False,
    ):
        super().__init__(gemini_api_key, model, constraint_response_schema)

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
        *,
        grid_divisions: Union[str, Path, os.PathLike],
        auxiliary_information: str | None = "",
        constraint_response_schema: bool | None = None,
        **kwargs,
    ) -> ImageBboxChallenge:
        model_to_use = kwargs.pop("model", self._model)
        if model_to_use is None:
            # Or raise an error, or use a default defined in this class if appropriate
            raise ValueError("Model must be provided either at initialization or via kwargs.")

        if constraint_response_schema is None:
            constraint_response_schema = self._constraint_response_schema

        enable_response_schema = kwargs.get("enable_response_schema")
        if enable_response_schema is not None:
            constraint_response_schema = enable_response_schema

        # Initialize Gemini client with API key
        client = genai.Client(api_key=self._api_key)

        # Upload the challenge image file
        files = await asyncio.gather(
            client.aio.files.upload(file=challenge_screenshot),
            client.aio.files.upload(file=grid_divisions),
        )

        # Create content with only the image
        parts = [
            types.Part.from_uri(file_uri=files[0].uri, mime_type=files[0].mime_type),
            types.Part.from_uri(file_uri=files[1].uri, mime_type=files[1].mime_type),
        ]
        if auxiliary_information and isinstance(auxiliary_information, str):
            parts.append(types.Part.from_text(text=auxiliary_information))

        contents = [types.Content(role="user", parts=parts)]

        system_instruction = SYSTEM_INSTRUCTIONS
        config = types.GenerateContentConfig(temperature=0, system_instruction=system_instruction)

        if model_to_use in ["gemini-2.5-flash-preview-04-17"]:
            config.thinking_config = types.ThinkingConfig(thinking_budget=0)

        # Change to JSON mode
        if not constraint_response_schema or model_to_use in [
            "gemini-2.0-flash-thinking-exp-01-21"
        ]:
            self._response = await client.aio.models.generate_content(
                model=model_to_use, contents=contents, config=config
            )

            return ImageBboxChallenge(**extract_first_json_block(self._response.text))

        config.response_mime_type = "application/json"
        config.response_schema = ImageBboxChallenge

        # Structured output with Constraint encoding
        self._response = await client.aio.models.generate_content(
            model=model_to_use, contents=contents, config=config
        )
        if _result := self._response.parsed:
            return ImageBboxChallenge(**self._response.parsed.model_dump())
        return ImageBboxChallenge(**extract_first_json_block(self._response.text))
