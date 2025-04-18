import os

import dotenv
from loguru import logger

from hcaptcha_challenger.tools import ImageClassifier

dotenv.load_dotenv()

gic = ImageClassifier(gemini_api_key=os.getenv("GEMINI_API_KEY"))


def test_gemini_image_classifier():
    screenshot_path = "challenge_view/image_label_binary/1.png"
    results = gic.invoke(
        screenshot_path, model="gemini-2.5-flash-preview-04-17", constraint_response_schema=True
    )
    logger.debug(f'ToolInvokeMessage: {results.log_message}')
