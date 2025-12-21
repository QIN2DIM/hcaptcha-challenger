# -*- coding: utf-8 -*-
"""
ChallengeRouter - Challenge type classification tool.

This module provides tools to classify challenge screenshots into their
respective types (single/multi select, single/multi drag).

Classes:
    ChallengeClassifier: Simple enum-based classification
    ChallengeRouter: Full classification with prompt extraction
"""
from pathlib import Path
from typing import Union

from google import genai
from google.genai import types
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_fixed

from hcaptcha_challenger.models import (
    FastShotModelType,
    ChallengeRouterResult,
    ChallengeTypeEnum,
    DEFAULT_FAST_SHOT_MODEL,
    THINKING_BUDGET_MODELS,
)
from hcaptcha_challenger.tools.internal.base import Reasoner
from hcaptcha_challenger.tools.internal.providers.gemini import extract_first_json_block
from hcaptcha_challenger.utils import load_desc

USER_PROMPT = """
Your task is to classify challenge questions into one of four types:
    - image_label_single_select (clicking ONE specific area/object)
    - image_label_multi_select (clicking MULTIPLE areas/objects)
    - image_drag_single (dragging ONE element/piece)
    - image_drag_multi (dragging MULTIPLE elements/pieces)
"""


class ChallengeClassifier(Reasoner[FastShotModelType, ChallengeTypeEnum]):
    """
    Simple challenge type classifier that returns an enum value.

    This is a lightweight classifier that only determines the challenge type
    without extracting the challenge prompt.
    """

    description: str = load_desc(Path(__file__).parent / "challenge_router.md")

    def __init__(
        self, gemini_api_key: str, model: FastShotModelType = DEFAULT_FAST_SHOT_MODEL, **kwargs
    ):
        super().__init__(gemini_api_key, model, **kwargs)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_fixed(3),
        before_sleep=lambda retry_state: logger.warning(
            f"Retry request ({retry_state.attempt_number}/3) - "
            f"Wait 3 seconds - Exception: {retry_state.outcome.exception()}"
        ),
    )
    async def __call__(
        self, *, challenge_screenshot: Union[str, Path], **kwargs
    ) -> ChallengeTypeEnum:
        """
        Classify a challenge screenshot into its type.

        Args:
            challenge_screenshot: Path to the challenge image.
            **kwargs: Additional options.

        Returns:
            ChallengeTypeEnum indicating the challenge type.
        """
        model_to_use = kwargs.pop("model", self._model)
        if model_to_use is None:
            raise ValueError("Model must be provided either at initialization or via kwargs.")

        client = genai.Client(api_key=self._api_key)
        files = [await client.aio.files.upload(file=challenge_screenshot)]

        # Handle models that don't support JSON response schema
        if model_to_use in ["gemini-2.0-flash-thinking-exp-01-21"]:
            contents = [
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_uri(file_uri=files[0].uri, mime_type=files[0].mime_type)
                    ],
                )
            ]
            response = await client.aio.models.generate_content(
                model=model_to_use,
                contents=contents,
                config=types.GenerateContentConfig(
                    temperature=0, system_instruction=self.description
                ),
            )
            self._response = response
            return ChallengeTypeEnum(response.text.strip())

        # Standard enum response
        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_uri(file_uri=files[0].uri, mime_type=files[0].mime_type),
                    types.Part.from_text(text=USER_PROMPT.strip()),
                ],
            )
        ]
        response = await client.aio.models.generate_content(
            model=model_to_use,
            contents=contents,
            config=types.GenerateContentConfig(
                temperature=0, response_mime_type="text/x.enum", response_schema=ChallengeTypeEnum
            ),
        )
        self._response = response
        return ChallengeTypeEnum(response.text.strip())


class ChallengeRouter(Reasoner[FastShotModelType, ChallengeRouterResult]):
    """
    Full challenge router that extracts both type and prompt.

    This classifier analyzes the challenge screenshot and returns both
    the challenge type and the extracted challenge prompt.
    """

    description: str = load_desc(Path(__file__).parent / "challenge_router.md")

    def __init__(
        self, gemini_api_key: str, model: FastShotModelType = DEFAULT_FAST_SHOT_MODEL, **kwargs
    ):
        super().__init__(gemini_api_key, model, **kwargs)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_fixed(3),
        before_sleep=lambda retry_state: logger.warning(
            f"Retry request ({retry_state.attempt_number}/3) - "
            f"Wait 3 seconds - Exception: {retry_state.outcome.exception()}"
        ),
    )
    async def __call__(
        self, *, challenge_screenshot: Union[str, Path], **kwargs
    ) -> ChallengeRouterResult:
        """
        Analyze a challenge screenshot and return type with prompt.

        Args:
            challenge_screenshot: Path to the challenge image.
            **kwargs: Additional options.

        Returns:
            ChallengeRouterResult with type and extracted prompt.
        """
        model_to_use = kwargs.pop("model", self._model)
        if model_to_use is None:
            raise ValueError("Model must be provided either at initialization or via kwargs.")

        client = genai.Client(api_key=self._api_key)
        files = [await client.aio.files.upload(file=challenge_screenshot)]

        parts = [
            types.Part.from_uri(file_uri=files[0].uri, mime_type=files[0].mime_type),
            types.Part.from_text(text=USER_PROMPT.strip()),
        ]
        contents = [types.Content(role="user", parts=parts)]

        config = types.GenerateContentConfig(
            temperature=0,
            response_mime_type="application/json",
            response_schema=ChallengeRouterResult,
        )

        if model_to_use in THINKING_BUDGET_MODELS and "pro" not in model_to_use:
            config.thinking_config = types.ThinkingConfig(include_thoughts=False)

        response = await client.aio.models.generate_content(
            model=model_to_use, contents=contents, config=config
        )
        self._response = response

        if response.parsed:
            return ChallengeRouterResult(**response.parsed.model_dump())
        return ChallengeRouterResult(**extract_first_json_block(response.text))
