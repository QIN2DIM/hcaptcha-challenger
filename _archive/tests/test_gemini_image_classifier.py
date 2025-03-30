import os

from loguru import logger

from hcaptcha_challenger.tools import GeminiImageClassifier

gic = GeminiImageClassifier(gemini_api_key=os.getenv("GEMINI_API_KEY"))


def test_gemini_image_classifier():
    screenshot_path = "challenge_view/image_label_binary/1.png"

    results = gic.invoke(screenshot_path, model="gemini-2.0-flash-thinking-exp-01-21")
    boolean_matrix = results.convert_box_to_boolean_matrix()

    logger.debug(f"Challenge Prompt: {results.challenge_prompt}")
    logger.debug(f"Coordinates: {results.coordinates}")
    logger.debug(f"Results: {boolean_matrix}")
