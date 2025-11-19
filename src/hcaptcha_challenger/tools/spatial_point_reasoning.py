import asyncio
import os
from pathlib import Path
from typing import Union

from google import genai
from google.genai import types
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_fixed

from hcaptcha_challenger.models import SCoTModelType, ImageAreaSelectChallenge, DEFAULT_SCOT_MODEL
from hcaptcha_challenger.tools.common import extract_first_json_block
from hcaptcha_challenger.tools.reasoner import _Reasoner

THINKING_PROMPT = """
**Rule for 'Find the Different Object' Tasks:**

*   **Constraint:** Do **NOT** consider size differences caused by perspective (near/far).
*   **Focus:** Identify difference based **only** on object outline, shape, and core structural features.

**Core Principles for Visual Analysis:**

*   **Processing Order:** Always analyze **Global Context** before **Local Details**.
*   **Perspective:** Maintain awareness of the overall scene ("look outside the immediate focus") when interpreting specific elements.
*   **Validation:** Ensure local interpretations are consistent with the global context to avoid settling for potentially incorrect "local optima".
*   **Method:** Employ a calm, systematic, top-down (Global-to-Local) analysis workflow.

**Workflow:**
1. Identify challenge prompt about the Challenge Image
2. Think about what the challenge requires identification goals, and where are they in the picture
3. Based on the plane rectangular coordinate system, reasoning about the absolute position of the "answer object" in the coordinate system

Finally, solve the challenge, locate the object, output the coordinates of the correct answer as json. 
"""


class SpatialPointReasoner(_Reasoner[SCoTModelType]):

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
    ) -> ImageAreaSelectChallenge:
        model_to_use = kwargs.pop("model", self._model)
        if model_to_use is None:
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
        # When the model performs inference, the image will also be converted into the corresponding Image Token.
        # When the context of a dialogue is long, the model may focus on the backward Prompt.
        # Therefore, when writing Prompt, you can say that the instructions are placed at the end
        # and the images are placed at the head, so that the model can pay more attention to the instructions,
        # thereby improving the effect of the instructions following.
        parts = []
        if challenge_screenshot:
            parts.append(types.Part.from_uri(file_uri=files[0].uri, mime_type=files[0].mime_type))
            parts.append(types.Part.from_uri(file_uri=files[1].uri, mime_type=files[1].mime_type))
        else:
            parts.append(types.Part.from_uri(file_uri=files[0].uri, mime_type=files[0].mime_type))
        if auxiliary_information and isinstance(auxiliary_information, str):
            parts.append(types.Part.from_text(text=auxiliary_information))

        contents = [types.Content(role="user", parts=parts)]

        config = types.GenerateContentConfig(
            system_instruction=THINKING_PROMPT,
            media_resolution=types.MediaResolution.MEDIA_RESOLUTION_HIGH,
            response_mime_type="application/json",
            response_schema=ImageAreaSelectChallenge,
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
            return ImageAreaSelectChallenge(**self._response.parsed.model_dump())
        return ImageAreaSelectChallenge(**extract_first_json_block(self._response.text))
