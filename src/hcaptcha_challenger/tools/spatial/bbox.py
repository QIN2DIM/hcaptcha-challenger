# -*- coding: utf-8 -*-
"""
SpatialBboxReasoner - Bounding box challenge solver.

This tool analyzes images to identify the precise bounding box coordinates
for the target area that fulfills the challenge requirements.
"""
from pathlib import Path
from typing import Union

from hcaptcha_challenger.models import ImageBboxChallenge
from hcaptcha_challenger.tools.spatial.base import SpatialReasoner
from hcaptcha_challenger.utils import load_desc


class SpatialBboxReasoner(SpatialReasoner[ImageBboxChallenge]):
    """
    Spatial bounding box reasoning tool for area identification challenges.

    Analyzes images to identify the precise bounding box coordinates
    for the target area based on the challenge requirements.

    Attributes:
        description: The system prompt for the tool.
    """

    description: str = load_desc(Path(__file__).parent / "bbox.md")

    async def __call__(
        self,
        *,
        challenge_screenshot: Union[str, Path],
        grid_divisions: Union[str, Path],
        auxiliary_information: str | None = None,
        **kwargs,
    ) -> ImageBboxChallenge:
        """
        Analyze a bounding box challenge and return the solution coordinates.

        Args:
            challenge_screenshot: Path to the challenge image.
            grid_divisions: Path to the grid overlay image.
            auxiliary_information: Optional challenge prompt or context.
            thinking_level: Thinking level for the model (default: HIGH).
            **kwargs: Additional options passed to the provider.

        Returns:
            ImageBboxChallenge containing the bounding box coordinates.
        """
        return await self._invoke_spatial(
            challenge_screenshot=Path(challenge_screenshot),
            grid_divisions=Path(grid_divisions),
            auxiliary_information=auxiliary_information,
            response_schema=ImageBboxChallenge,
            **kwargs,
        )
