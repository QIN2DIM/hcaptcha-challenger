# -*- coding: utf-8 -*-
"""
SpatialPathReasoner - Drag and drop challenge solver.

This tool analyzes images to identify which draggable element should be
moved to which target location based on visual patterns and implicit matching rules.
"""
from pathlib import Path
from typing import Union

from hcaptcha_challenger.models import ImageDragDropChallenge
from hcaptcha_challenger.tools.spatial.base import SpatialReasoner
from hcaptcha_challenger.utils import load_desc


class SpatialPathReasoner(SpatialReasoner[ImageDragDropChallenge]):
    """
    Spatial path reasoning tool for drag and drop challenges.

    Analyzes images to identify the correct drag-and-drop paths based on
    visual patterns and implicit matching rules.

    Attributes:
        description: The system prompt for the tool.
    """

    description: str = load_desc(Path(__file__).parent / "path.md")

    async def __call__(
        self,
        *,
        challenge_screenshot: Union[str, Path],
        grid_divisions: Union[str, Path],
        auxiliary_information: str | None = None,
        **kwargs,
    ) -> ImageDragDropChallenge:
        """
        Analyze a drag-and-drop challenge and return the solution paths.

        Args:
            challenge_screenshot: Path to the challenge image.
            grid_divisions: Path to the grid overlay image.
            auxiliary_information: Optional challenge prompt or context.
            thinking_level: Thinking level for the model (default: HIGH).
            **kwargs: Additional options passed to the provider.

        Returns:
            ImageDragDropChallenge containing the drag paths.
        """
        return await self._invoke_spatial(
            challenge_screenshot=Path(challenge_screenshot),
            grid_divisions=Path(grid_divisions),
            auxiliary_information=auxiliary_information,
            response_schema=ImageDragDropChallenge,
            **kwargs,
        )
