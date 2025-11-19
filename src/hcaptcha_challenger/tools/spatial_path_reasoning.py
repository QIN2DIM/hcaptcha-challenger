import os
from pathlib import Path
from typing import Union, List

from google import genai
from google.genai import types
from hcaptcha_challenger.models import SCoTModelType, ImageDragDropChallenge
from hcaptcha_challenger.tools.reasoner import _Reasoner


SYSTEM_INSTRUCTION = """
You are a Visual Spatial Reasoning System specialized in solving interactive placement puzzles.

Your task: Analyze the image to identify which draggable element should be moved to which target location based on visual patterns and implicit matching rules.

Key capabilities:
- Recognize spatial relationships between objects across the canvas
- Identify visual patterns (shape similarity, property matching, categorical grouping)
- Infer implicit rules without explicit instructions
- Map source elements to their corresponding target positions

Critical: The image contains a coordinate system with labeled axes (X Coordinate, Y Coordinate). Read coordinates directly from these axis scales, NOT from image pixel positions.

Output your solution as structured coordinates identifying the movement path.
"""

AUXILIARY_INFORMATION_TPL = """
**Challenge Prompt:**
{auxiliary_information}
"""

USER_PROMPT = """
Analyze the visual puzzle:
- Identify the draggable element and available target zones
- Recognize the matching pattern (visual similarity, categorical logic, or spatial rules)
- Determine the correct target position for the draggable element
- Read coordinates from the labeled coordinate axes shown in the image (not pixel positions)
- Provide precise x,y values for both source and destination based on the axis scales
"""


class SpatialPathReasoner(_Reasoner[SCoTModelType]):

    async def _draw_thoughts_parts(
        self,
        client: genai.Client,
        grid_divisions: Union[str, Path, os.PathLike],
        auxiliary_information: str,
        challenge_screenshot: Union[str, Path, os.PathLike] | None = None,
    ) -> List[types.Part]:
        files_to_upload = [challenge_screenshot, grid_divisions]
        uploaded_files = await self._upload_files(client, files_to_upload)
        parts = self._files_to_parts(uploaded_files)

        if auxiliary_information and isinstance(auxiliary_information, str):
            ait = AUXILIARY_INFORMATION_TPL.format(auxiliary_information=auxiliary_information)
            parts.append(types.Part.from_text(text=f"{ait}{USER_PROMPT}"))
        else:
            parts.append(types.Part.from_text(text=USER_PROMPT))

        return parts

    async def invoke_async(
        self,
        *,
        challenge_screenshot: Union[str, Path, os.PathLike],
        grid_divisions: Union[str, Path, os.PathLike],
        auxiliary_information: str | None = "",
        **kwargs,
    ) -> ImageDragDropChallenge:
        model_to_use = kwargs.pop("model", self._model)
        if model_to_use is None:
            raise ValueError("Model must be provided either at initialization or via kwargs.")

        client = genai.Client(api_key=self._api_key)

        parts = await self._draw_thoughts_parts(
            client=client,
            challenge_screenshot=challenge_screenshot,
            grid_divisions=grid_divisions,
            auxiliary_information=auxiliary_information,
        )

        contents = [types.Content(role="user", parts=parts)]

        config = types.GenerateContentConfig(
            temperature=0,
            system_instruction=SYSTEM_INSTRUCTION,
            media_resolution=types.MediaResolution.MEDIA_RESOLUTION_HIGH,
            response_mime_type="application/json",
            response_schema=ImageDragDropChallenge,
        )

        self._set_thinking_config(
            config=config,
            model_to_use=model_to_use,
            thinking_level=kwargs.get("thinking_level", types.ThinkingLevel.HIGH),
        )

        return await self._generate_content(
            client=client,
            model=model_to_use,
            contents=contents,
            config=config,
            response_schema=ImageDragDropChallenge,
        )
