import os
from pathlib import Path
from typing import Union

from google import genai
from google.genai import types

from hcaptcha_challenger.models import SCoTModelType, ImageBinaryChallenge
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
Solve the challenge, use [0,0] ~ [2,2] to locate 9grid, output the coordinates of the correct answer as JSON.
"""


class ImageClassifier(_Reasoner[SCoTModelType]):
    """
    A classifier that uses Google's Gemini AI models to analyze and solve image-based challenges.

    This class provides functionality to process screenshots of binary image challenges
    (typically grid-based selection challenges) and determines the correct answer coordinates.
    """

    async def invoke_async(
        self, *, challenge_screenshot: Union[str, Path, os.PathLike], **kwargs
    ) -> ImageBinaryChallenge:
        """
        Process an image challenge and return the solution coordinates.

        Args:
            challenge_screenshot: The image file containing the challenge to solve

        Returns:
            ImageBinaryChallenge: Object containing the solution coordinates
        """
        model_to_use = kwargs.pop("model", self._model)
        if model_to_use is None:
            # Or raise an error, or use a default defined in this class if appropriate
            raise ValueError("Model must be provided either at initialization or via kwargs.")

        # Initialize Gemini client with API_KEY
        client = genai.Client(api_key=self._api_key)

        # Upload the challenge image file
        files_to_upload = [challenge_screenshot]
        uploaded_files = await self._upload_files(client, files_to_upload)

        parts = self._files_to_parts(uploaded_files)

        # Handle models that support JSON response schema
        parts.append(types.Part.from_text(text=USER_PROMPT.strip()))

        contents = [types.Content(role="user", parts=parts)]

        config = types.GenerateContentConfig(
            temperature=0,
            system_instruction=SYSTEM_INSTRUCTION,
            media_resolution=types.MediaResolution.MEDIA_RESOLUTION_HIGH,
            response_mime_type="application/json",
            response_schema=ImageBinaryChallenge,
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
            response_schema=ImageBinaryChallenge,
        )
