import asyncio
import os
from pathlib import Path
from typing import Union, List

from google import genai
from google.genai import types
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_fixed

from hcaptcha_challenger.models import SCoTModelType, ImageDragDropChallenge, DEFAULT_SCOT_MODEL
from hcaptcha_challenger.tools.common import extract_first_json_block
from hcaptcha_challenger.tools.reasoner import _Reasoner

THINKING_PROMPT = """
**Rule for 'Find the Notched Rectangular Area' Tasks:**
1. Identify challenge prompt about the Challenge Image
2. Think about what the challenge requires identification goals, and where are they in the picture
3. Think about what object should be dragged to which position
4. Based on the plane rectangular coordinate system, reasoning about the absolute position of the "answer object" in the coordinate system

Finally, solve the challenge, locate the object, output the coordinates of the correct answer as json. Follow the following format to return a coordinates wrapped with a json code block:

```json
{
  "challenge_prompt": "Task description",
  "paths": [
    {"start_point": {"x":  x1, "y": y1}, "end_point": {"x":  x2, "y": y2}}
  ]
}
```
"""

USER_PROMPT = """
请根据教程学习规则、模式和思路，尝试解决新的 image_drag_drop challenge，最后返回正确答案的坐标。
将右侧的拼图方块移至左侧画布上的正确位置，使得画布上的物体形状完整和连续。
"""

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


async def draw_speculative_sampling_parts(
    client: genai.Client,
    challenge_screenshot: Union[str, Path, os.PathLike],
    grid_divisions: Union[str, Path, os.PathLike],
    auxiliary_information: str,
) -> List[types.Part] | None:
    scot_dir = Path(__file__).parent.joinpath("scot")

    files = await asyncio.gather(
        client.aio.files.upload(file=scot_dir.joinpath("image_drag_drop_few_shot_001.png")),
        client.aio.files.upload(file=scot_dir.joinpath("image_drag_drop_few_shot_002.png")),
        client.aio.files.upload(file=challenge_screenshot),
        client.aio.files.upload(file=grid_divisions),
    )

    parts = [
        types.Part.from_uri(file_uri=files[0].uri, mime_type=files[0].mime_type),
        types.Part.from_uri(file_uri=files[1].uri, mime_type=files[1].mime_type),
        types.Part.from_uri(file_uri=files[2].uri, mime_type=files[2].mime_type),
        types.Part.from_uri(file_uri=files[3].uri, mime_type=files[3].mime_type),
    ]

    user_prompt = USER_PROMPT
    if auxiliary_information and isinstance(auxiliary_information, str):
        user_prompt += f"\n{auxiliary_information}"
    parts.append(types.Part.from_text(text=user_prompt))

    logger.debug(f"User prompt: {user_prompt}")
    return parts


async def draw_thoughts_parts(
    client: genai.Client,
    challenge_screenshot: Union[str, Path, os.PathLike],
    grid_divisions: Union[str, Path, os.PathLike],
    auxiliary_information: str,
) -> List[types.Part]:
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
        ait = AUXILIARY_INFORMATION_TPL.format(auxiliary_information=auxiliary_information)
        parts.append(types.Part.from_text(text=f"{ait}{USER_PROMPT_1022}"))
    else:
        parts.append(types.Part.from_text(text=USER_PROMPT_1022))

    return parts


class SpatialPathReasoner(_Reasoner[SCoTModelType]):

    def __init__(
        self,
        gemini_api_key: str,
        model: SCoTModelType = DEFAULT_SCOT_MODEL,
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
    ) -> ImageDragDropChallenge:
        model_to_use = kwargs.pop("model", self._model)
        if model_to_use is None:
            raise ValueError("Model must be provided either at initialization or via kwargs.")

        if constraint_response_schema is None:
            constraint_response_schema = self._constraint_response_schema

        enable_response_schema = kwargs.get("enable_response_schema")
        if enable_response_schema is not None:
            constraint_response_schema = enable_response_schema

        enable_scot = False

        system_instruction = THINKING_PROMPT_1022

        # Initialize Gemini client with API key
        client = genai.Client(api_key=self._api_key)

        if enable_scot and model_to_use not in ["gemini-2.0-flash-thinking-exp-01-21"]:
            parts = await draw_speculative_sampling_parts(
                client, challenge_screenshot, grid_divisions, auxiliary_information
            )
            constraint_response_schema = True
            system_instruction = None
        else:
            parts = await draw_thoughts_parts(
                client, challenge_screenshot, grid_divisions, auxiliary_information
            )

        contents = [types.Content(role="user", parts=parts)]

        config = types.GenerateContentConfig(
            temperature=kwargs.get("temperature", 0),
            system_instruction=system_instruction,
            thinking_config=self._pin_thinking_config(
                model_to_use=model_to_use, thinking_budget=kwargs.get("thinking_budget")
            ),
        )

        # Change to JSON mode
        if not constraint_response_schema or model_to_use in [
            "gemini-2.0-flash-thinking-exp-01-21"
        ]:
            self._response = await client.aio.models.generate_content(
                model=model_to_use, contents=contents, config=config
            )
            return ImageDragDropChallenge(**extract_first_json_block(self._response.text))

        # Structured output with Constraint encoding
        config.response_mime_type = "application/json"
        config.response_schema = ImageDragDropChallenge

        self._response = await client.aio.models.generate_content(
            model=model_to_use, contents=contents, config=config
        )
        if _result := self._response.parsed:
            return ImageDragDropChallenge(**self._response.parsed.model_dump())
        return ImageDragDropChallenge(**extract_first_json_block(self._response.text))
