# -*- coding: utf-8 -*-
"""
SpatialReasoner - Base class for spatial reasoning tools.

This intermediate base class provides common functionality for all
spatial reasoning tools (path, point, bbox).
"""
from abc import ABC
from pathlib import Path
from typing import TypeVar, List

from pydantic import BaseModel

from hcaptcha_challenger.models import SCoTModelType
from hcaptcha_challenger.tools.internal.base import Reasoner

ResponseT = TypeVar("ResponseT", bound=BaseModel)


class SpatialReasoner(Reasoner[SCoTModelType, ResponseT], ABC):
    """
    Base class for spatial reasoning tools.

    Provides common defaults for spatial reasoning:
    - Higher temperature (1.0) for creative spatial reasoning
    - High thinking level by default
    - Standard image upload pattern (challenge + grid)
    """

    async def _invoke_spatial(
        self,
        *,
        challenge_screenshot: Path,
        grid_divisions: Path,
        auxiliary_information: str | None = None,
        response_schema: type[ResponseT],
        **kwargs,
    ) -> ResponseT:
        """
        Common invocation logic for spatial reasoning.

        Args:
            challenge_screenshot: Path to the challenge image.
            grid_divisions: Path to the grid overlay image.
            auxiliary_information: Optional user prompt with additional context.
            thinking_level: Override for thinking level.
            response_schema: Pydantic model for structured output.
            temperature: Override for sampling temperature.
            **kwargs: Additional provider options.

        Returns:
            Parsed response matching the response_schema.
        """
        images: List[Path] = [challenge_screenshot, grid_divisions]

        return await self._provider.generate_with_images(
            images=images,
            user_prompt=auxiliary_information,
            description=self.description,
            response_schema=response_schema,
            **kwargs,
        )
