# -*- coding: utf-8 -*-
import os
from pathlib import Path

import dotenv
import pytest

from hcaptcha_challenger import ImageClassifier
from hcaptcha_challenger.tools.internal.providers import OpenAICompatibleProvider

dotenv.load_dotenv()

CHALLENGE_VIEW_DIR = Path(__file__).parent.joinpath("challenge_view/image_label_binary")

pytestmark = pytest.mark.skipif(
    not (os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_KEY")),
    reason="no OpenAI-compatible endpoint configured",
)


async def test_openai_image_classifier():
    provider = OpenAICompatibleProvider(
        model=os.getenv("OPENAI_VISION_MODEL", "qwen2-vl:7b"),
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
    )
    ic = ImageClassifier(provider=provider)
    results = await ic(challenge_screenshot=CHALLENGE_VIEW_DIR.joinpath("1.png"))
    assert results is not None
