# -*- coding: utf-8 -*-
"""
ImageClassifier - 9-grid image classification challenge solver.

This tool analyzes 9-grid challenge screenshots and identifies which
cells should be selected based on the challenge prompt.
"""
from pathlib import Path
from typing import Union

from hcaptcha_challenger.models import SCoTModelType, ImageBinaryChallenge, DEFAULT_SCOT_MODEL
from hcaptcha_challenger.tools.internal.base import Reasoner
from hcaptcha_challenger.tools.internal.providers.protocol import ChatProvider
from hcaptcha_challenger.utils import load_desc


class ImageClassifier(Reasoner[SCoTModelType, ImageBinaryChallenge]):
    """
    A classifier for 9-grid image selection challenges.

    This tool processes screenshots of binary image challenges (typically
    3x3 grid-based selection challenges) and determines which cells contain
    the correct answer.

    The grid uses [row, col] coordinates where:
    - [0,0] is top-left, [0,2] is top-right
    - [2,0] is bottom-left, [2,2] is bottom-right

    Attributes:
        description: The system prompt for the tool.
    """

    description: str = load_desc(Path(__file__).parent / "image_classifier.md")

    def __init__(
        self,
        gemini_api_key: str,
        model: SCoTModelType = DEFAULT_SCOT_MODEL,
        *,
        provider: ChatProvider | None = None,
        **kwargs,
    ):
        super().__init__(gemini_api_key, model, provider=provider, **kwargs)

    async def __call__(
        self, *, challenge_screenshot: Union[str, Path], **kwargs
    ) -> ImageBinaryChallenge:
        """
        Analyze a 9-grid challenge and return the solution coordinates.

        Args:
            challenge_screenshot: Path to the challenge image.
            **kwargs: Additional options passed to the provider.

        Returns:
            ImageBinaryChallenge containing the selected cell coordinates.
        """
        return await self._provider.generate_with_images(
            images=[Path(challenge_screenshot)],
            user_prompt="Solve the challenge. Identify the correct images in the 3x3 grid. Output the coordinates of the correct answer as JSON. IMPORTANT: Each coordinate MUST be a list of two integers [row, col], e.g., [0, 0] for top-left, [1, 2] for middle-right.",
            description=self.description,
            response_schema=ImageBinaryChallenge,
            **kwargs,
        )
