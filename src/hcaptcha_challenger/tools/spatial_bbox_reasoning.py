from pathlib import Path
from typing import Union
import os

from google import genai
from google.genai import types

from hcaptcha_challenger.models import SCoTModelType, ImageBboxChallenge
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

    async def invoke_async(
        self,
        *,
        challenge_screenshot: Union[str, Path, os.PathLike],
        grid_divisions: Union[str, Path, os.PathLike],
        auxiliary_information: str | None = "",
        **kwargs,
    ) -> ImageBboxChallenge:
        model_to_use = kwargs.pop("model", self._model)
        if model_to_use is None:
            raise ValueError("Model must be provided either at initialization or via kwargs.")

        client = genai.Client(api_key=self._api_key)

        files_to_upload = [challenge_screenshot, grid_divisions]
        uploaded_files = await self._upload_files(client, files_to_upload)

        parts = self._files_to_parts(uploaded_files)

        if auxiliary_information and isinstance(auxiliary_information, str):
            parts.append(types.Part.from_text(text=auxiliary_information))

        contents = [types.Content(role="user", parts=parts)]

        config = types.GenerateContentConfig(
            temperature=0,
            system_instruction=SYSTEM_INSTRUCTIONS,
            media_resolution=types.MediaResolution.MEDIA_RESOLUTION_HIGH,
            response_mime_type="application/json",
            response_schema=ImageBboxChallenge,
        )

        self._set_thinking_config(
            config=config,
            model_to_use=model_to_use,
            thinking_level=kwargs.get("thinking_level", types.ThinkingLevel.LOW),
        )

        return await self._generate_content(
            client=client,
            model=model_to_use,
            contents=contents,
            config=config,
            response_schema=ImageBboxChallenge,
        )
