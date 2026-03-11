# -*- coding: utf-8 -*-
"""
SpatialPointReasoner - Image area selection challenge solver.

This tool analyzes images to identify specific points or areas that match
the challenge requirements, using global-to-local visual analysis.
"""
from pathlib import Path
from typing import Union

from hcaptcha_challenger.models import ImageAreaSelectChallenge
from hcaptcha_challenger.tools.spatial.base import SpatialReasoner
from hcaptcha_challenger.utils import load_desc


class SpatialPointReasoner(SpatialReasoner[ImageAreaSelectChallenge]):
    """
    Spatial point reasoning tool for image area selection challenges.

    Uses a systematic global-to-local analysis workflow to identify
    the correct click coordinates based on the challenge prompt.

    Attributes:
        description: The system prompt for the tool.
    """

    description: str = load_desc(Path(__file__).parent / "point.md")

    async def __call__(
        self,
        *,
        challenge_screenshot: Union[str, Path, list[Path]],
        grid_divisions: Union[str, Path],
        auxiliary_information: str | None = None,
        **kwargs,
    ) -> ImageAreaSelectChallenge:
        """
        Analyze an area selection challenge and return the solution points.

        Args:
            challenge_screenshot: Path(s) to the challenge image(s).
            grid_divisions: Path to the grid overlay image.
            auxiliary_information: Optional challenge prompt or context.
            thinking_level: Thinking level for the model (default: HIGH).
            **kwargs: Additional options passed to the provider.

        Returns:
            ImageAreaSelectChallenge containing the click coordinates.
        """
        if isinstance(challenge_screenshot, list):
             cs = [Path(p) for p in challenge_screenshot]
        else:
             cs = Path(challenge_screenshot)

        return await self._invoke_spatial(
            challenge_screenshot=cs,
            grid_divisions=Path(grid_divisions),
            auxiliary_information=auxiliary_information,
            response_schema=ImageAreaSelectChallenge,
            **kwargs,
        )
