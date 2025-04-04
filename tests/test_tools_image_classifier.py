import os

import dotenv
from loguru import logger

from hcaptcha_challenger.tools import ImageClassifier

dotenv.load_dotenv()

gic = ImageClassifier(gemini_api_key=os.getenv("GEMINI_API_KEY"))


def test_gemini_image_classifier():
    screenshot_path = "challenge_view/image_label_binary/1.png"
    results = gic.invoke(screenshot_path, model="gemini-2.5-pro-preview-03-25")
    logger.debug(f'ToolInvokeMessage: {results.log_message}')
