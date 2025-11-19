import os
from pathlib import Path
from typing import Union

from google import genai
from google.genai import types

from hcaptcha_challenger.models import SCoTModelType, ImageAreaSelectChallenge
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

    async def invoke_async(
        self,
        *,
        challenge_screenshot: Union[str, Path, os.PathLike],
        grid_divisions: Union[str, Path, os.PathLike],
        auxiliary_information: str | None = "",
        **kwargs,
    ) -> ImageAreaSelectChallenge:
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
            system_instruction=THINKING_PROMPT,
            media_resolution=types.MediaResolution.MEDIA_RESOLUTION_HIGH,
            response_mime_type="application/json",
            response_schema=ImageAreaSelectChallenge,
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
            response_schema=ImageAreaSelectChallenge,
        )
