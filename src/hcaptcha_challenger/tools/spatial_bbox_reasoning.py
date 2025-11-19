import asyncio
import os
from pathlib import Path
from typing import Union

from google import genai
from google.genai import types
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_fixed

from hcaptcha_challenger.models import SCoTModelType, ImageBboxChallenge, DEFAULT_SCOT_MODEL
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

    def __init__(self, gemini_api_key: str, model: SCoTModelType = DEFAULT_SCOT_MODEL, **kwargs):
        super().__init__(gemini_api_key, model, **kwargs)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_fixed(3),
        before_sleep=lambda retry_state: logger.warning(
            f"Retry request ({retry_state.attempt_number}/2) - Wait 3 seconds - Exception: {retry_state.outcome.exception()}"
        ),
    )
    async def invoke_async(
        self,
        *,
        grid_divisions: Union[str, Path, os.PathLike],
        challenge_screenshot: Union[str, Path, os.PathLike] | None = None,
        auxiliary_information: str | None = "",
        **kwargs,
    ) -> ImageBboxChallenge:
        model_to_use = kwargs.pop("model", self._model)
        if model_to_use is None:
            # Or raise an error, or use a default defined in this class if appropriate
            raise ValueError("Model must be provided either at initialization or via kwargs.")

        # Initialize Gemini client with API key
        client = genai.Client(api_key=self._api_key)

        # Upload the challenge image file
        upload_tasks = []
        if challenge_screenshot:
            upload_tasks.append(client.aio.files.upload(file=challenge_screenshot))
        upload_tasks.append(client.aio.files.upload(file=grid_divisions))

        files = await asyncio.gather(*upload_tasks)

        # Create content with only the image
        parts = []
        if challenge_screenshot:
            parts.append(types.Part.from_uri(file_uri=files[0].uri, mime_type=files[0].mime_type))
            parts.append(types.Part.from_uri(file_uri=files[1].uri, mime_type=files[1].mime_type))
        else:
            parts.append(types.Part.from_uri(file_uri=files[0].uri, mime_type=files[0].mime_type))
        if auxiliary_information and isinstance(auxiliary_information, str):
            parts.append(types.Part.from_text(text=auxiliary_information))

        contents = [types.Content(role="user", parts=parts)]

        system_instruction = SYSTEM_INSTRUCTIONS

        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            media_resolution=types.MediaResolution.MEDIA_RESOLUTION_HIGH,
            response_mime_type="application/json",
            response_schema=ImageBboxChallenge,
        )

        self._set_temperature(config=config, model_to_use=model_to_use)

        self._set_thinking_config(
            config=config,
            model_to_use=model_to_use,
            thinking_level=kwargs.get("thinking_level", types.ThinkingLevel.LOW),
        )

        # Structured output with Constraint encoding
        self._response = await client.aio.models.generate_content(
            model=model_to_use, contents=contents, config=config
        )
        if _result := self._response.parsed:
            return ImageBboxChallenge(**self._response.parsed.model_dump())
        return ImageBboxChallenge(**extract_first_json_block(self._response.text))
