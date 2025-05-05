import os
from pathlib import Path

import dotenv
import pytest
from loguru import logger

from hcaptcha_challenger.tools import ImageClassifier

dotenv.load_dotenv()

gic = ImageClassifier(gemini_api_key=os.getenv("GEMINI_API_KEY"))

CHALLENGE_VIEW_DIR = Path(__file__).parent.joinpath("challenge_view/image_label_binary")


def test_gemini_image_classifier():
    screenshot_path = CHALLENGE_VIEW_DIR.joinpath("1.png")
    results = gic.invoke(screenshot_path, model="gemini-2.5-flash-preview-04-17")
    logger.debug(f'ToolInvokeMessage: {results.log_message}')
