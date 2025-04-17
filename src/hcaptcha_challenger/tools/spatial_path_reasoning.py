import os
from pathlib import Path
from typing import Union

from google import genai
from google.genai import types
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_fixed

from hcaptcha_challenger.models import SCoTModelType, ImageDragDropChallenge
from hcaptcha_challenger.tools.common import extract_first_json_block

THINKING_PROMPT = """
<instructions>
Solve the visual challenge by accurately dragging the provided piece to complete the main shape.
Learn the rules of the game and understand what an endpoint is and the correct answer based on a reference video provided by the user.
</instructions>

<challenge_analysis>

1.  **Understand the Task:** The instruction requires dragging the 'missing piece' to fill a specific gap in the 'shape' presented on the left, making the shape whole.
2.  **Identify the Movable Piece:** Locate the distinct, separate image segment (often contained within a box or highlighted area). This is the 'piece' that needs to be moved.
3.  **Identify the Target Location (Gap):** Carefully examine the main shape on the left. Identify the precise gap, notch, or missing section where the 'piece' is clearly intended to fit based on its contours and the surrounding shape. The 'piece' should visually complete the shape when placed correctly in this specific location. This specific location is the target 'gap'.
4.  **Determine Coordinates:**
    *   Using the provided coordinate grid (or estimating based on image dimensions), determine the approximate center coordinates (x, y) of the 'piece' in its initial position. This is the `start_point`.
    *   Determine the approximate center coordinates (x, y) of the target 'gap' on the main shape where the piece needs to be placed to fit perfectly. This is the `end_point`. The goal is to align the center of the piece with the center of the gap.
5.  **Formulate the Path:** Define the drag-and-drop operation by specifying the path from the `start_point` (center of the 80x80 piece) to the `end_point` (center of the 80x80 target gap).

</challenge_analysis>


<output>
Provide the solution as a JSON object containing the challenge prompt description and the calculated path(s). Adhere strictly to the following format:

```json
{
  "challenge_prompt": "task description",
  "paths": [
    {"start_point": {"x": piece_center_x, "y": piece_center_y}, "end_point": {"x": gap_center_x, "y": gap_center_y}}
  ]
}
```
</output>
"""

SCOT_VIDEO_PATH = Path(__file__).parent.joinpath("scot/scot_guideline.mp4")


class SpatialPathReasoner:
    def __init__(self, gemini_api_key: str):
        """Initialize the classifier with a Gemini API key."""
        self._scot_file_uri = ""
        self._scot_file_mime_type = "video/mp4"
        self._client = genai.Client(api_key=gemini_api_key)

    @property
    def scot_file_uri(self) -> str:
        if not self._scot_file_uri:
            file = self._client.files.upload(file=SCOT_VIDEO_PATH)
            self._scot_file_uri = file.uri
            self._scot_file_mime_type = file.mime_type
        return self._scot_file_uri

    def insert_scot_parts_by_remote(self):
        scot_parts = []

        try:
            scot_parts.append(
                types.Part.from_uri(
                    file_uri=self.scot_file_uri, mime_type=self._scot_file_mime_type
                )
            )
        except Exception as e:
            logger.error(f"Error loading SCOT parts: {e}")

        for p in scot_parts:
            logger.debug(f"Insert SCOT Parts - {p=}")

        return scot_parts

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
        *,
        enable_response_schema: bool = False,
        enable_scot: bool = False,
        **kwargs,
    ) -> ImageDragDropChallenge:
        parts = []
        # {{< Insert Spatial Chain-of-Thought Parts >}}
        if enable_scot:
            parts.extend(self.insert_scot_parts_by_remote())

        # {{< Insert Challenge Parts >}}
        for image in [challenge_screenshot, grid_divisions]:
            file = self._client.files.upload(file=image)
            parts.append(types.Part.from_uri(file_uri=file.uri, mime_type=file.mime_type))

        # {{< Insert Custom User Prompt >}}
        if auxiliary_information and isinstance(auxiliary_information, str):
            parts.append(types.Part.from_text(text=auxiliary_information))

        # {{< Merge ALL Parts >}}
        contents = [types.UserContent(parts=parts)]

        # Change to JSON mode
        if not enable_response_schema or model in ["gemini-2.0-flash-thinking-exp-01-21"]:
            response = self._client.models.generate_content(
                model=model,
                contents=contents,
                config=types.GenerateContentConfig(
                    temperature=0, system_instruction=THINKING_PROMPT
                ),
            )
            return ImageDragDropChallenge(**extract_first_json_block(response.text))

        # Structured output with Constraint encoding
        response = self._client.models.generate_content(
            model=model,
            contents=contents,
            config=types.GenerateContentConfig(
                temperature=0,
                system_instruction=THINKING_PROMPT,
                response_mime_type="application/json",
                response_schema=ImageDragDropChallenge,
            ),
        )
        if _result := response.parsed:
            return ImageDragDropChallenge(**response.parsed.model_dump())
        return ImageDragDropChallenge(**extract_first_json_block(response.text))
