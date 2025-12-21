# -*- coding: utf-8 -*-
"""
SpatialPathReasoner - Drag and drop challenge solver.

This tool analyzes images to identify which draggable element should be
moved to which target location based on visual patterns and implicit matching rules.
"""
from pathlib import Path
from typing import Union

from google.genai import types

from hcaptcha_challenger.models import ImageDragDropChallenge
from hcaptcha_challenger.tools.spatial.base import SpatialReasoner
from hcaptcha_challenger.utils import load_desc

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


class SpatialPathReasoner(SpatialReasoner[ImageDragDropChallenge]):
    """
    Spatial path reasoning tool for drag and drop challenges.

    Analyzes images to identify the correct drag-and-drop paths based on
    visual patterns and implicit matching rules.

    Attributes:
        description: The system prompt for the tool.
    """

    description: str = load_desc(Path(__file__).parent / "path.md")

    async def invoke_async(
        self,
        *,
        challenge_screenshot: Union[str, Path],
        grid_divisions: Union[str, Path],
        auxiliary_information: str | None = None,
        thinking_level: types.ThinkingLevel | None = types.ThinkingLevel.HIGH,
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
        # Build user prompt with auxiliary information
        user_prompt = USER_PROMPT.strip()
        if auxiliary_information:
            user_prompt = (
                AUXILIARY_INFORMATION_TPL.format(auxiliary_information=auxiliary_information)
                + user_prompt
            )

        return await self._invoke_spatial(
            challenge_screenshot=Path(challenge_screenshot),
            grid_divisions=Path(grid_divisions),
            auxiliary_information=user_prompt,
            thinking_level=thinking_level,
            response_schema=ImageDragDropChallenge,
            **kwargs,
        )
