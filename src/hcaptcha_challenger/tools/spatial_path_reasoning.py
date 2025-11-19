import os
from pathlib import Path
from typing import Union, List

from google import genai
from google.genai import types
from hcaptcha_challenger.models import SCoTModelType, ImageDragDropChallenge
from hcaptcha_challenger.tools.reasoner import _Reasoner


THINKING_PROMPT_1022 = """
You are an expert-level Visual Puzzle Analyst and Logic Inference Engine. Your primary mission is to analyze images containing challenges and determine a solution that involves identifying a "source" object and a "destination" location.

You must follow these core principles for every task:

1.  **Deconstruct the Goal:** First, meticulously analyze the provided text instruction to understand the explicit goal of the challenge.
2.  **Identify Key Elements:** Scan the entire image to identify the key visual elements:
    *   The **Source Object**: The item that needs to be moved or placed.
    *   The **Destination Area**: The game board, grid, or context where the object should be placed.
    *   **Contextual Clues**: All other elements on the board that will be used to infer the rules.
3.  **Infer the Rules (Most Critical Step):** The rules of the puzzle are NOT given to you. You MUST deduce them by identifying patterns, sequences, logical groupings, or principles of exclusion from the contextual clues. State the rule you have inferred clearly.
4.  **Reason Step-by-Step:** Externalize your entire thought process. Follow a clear, logical sequence from goal analysis to final solution. Do not jump to conclusions.
5.  **Output in Structured Format:** Provide your final answer in a strict JSON format, specifying the source and destination coordinates.

Your entire process is about inferring hidden rules from visual data to satisfy a given textual goal.
"""

AUXILIARY_INFORMATION_TPL = """
**Challenge Prompt:**
{auxiliary_information}
"""

USER_PROMPT_1022 = """
**Your Analysis:**
Please follow your core principles and provide your step-by-step reasoning below to solve this challenge.

1.  **Goal Analysis:** Based on the Challenge Prompt, what is my primary objective?
2.  **Source Identification:** Describe and locate the 'Source Object' that needs to be moved.
3.  **Destination Area Identification:** Describe the area where the Source Object must be placed.
4.  **Rule Inference:**
    *   Observe the patterns in the Destination Area. What are the logical rules governing the placement of objects? (e.g., column-based categories, row-based sequences, color matching, shape exclusion, etc.)
    *   State the inferred rule clearly.

5.  **Solution Determination:** Applying the inferred rule, where is the exact 'correct location' for the Source Object? Based on the plane rectangular coordinate system, reasoning about the absolute position of the 'correct location' in the coordinate system.
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
            parts.append(types.Part.from_text(text=f"{ait}{USER_PROMPT_1022}"))
        else:
            parts.append(types.Part.from_text(text=USER_PROMPT_1022))

        return parts

    async def invoke_async(
        self,
        *,
        grid_divisions: Union[str, Path, os.PathLike],
        challenge_screenshot: Union[str, Path, os.PathLike] | None = None,
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
            system_instruction=THINKING_PROMPT_1022,
            media_resolution=types.MediaResolution.MEDIA_RESOLUTION_HIGH,
            response_mime_type="application/json",
            response_schema=ImageDragDropChallenge,
        )

        self._set_temperature(config=config, model_to_use=model_to_use)

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
            response_schema=ImageDragDropChallenge,
        )
