import os

import dotenv

from hcaptcha_challenger.tools.challenge_classifier import ChallengeClassifier, ChallengeTypeEnum

dotenv.load_dotenv()

gic = ChallengeClassifier(gemini_api_key=os.getenv("GEMINI_API_KEY"))


def test_challenge_classifier_image_drag_drop():
    screenshot_path = "challenge_view/image_drag_drop/single_1.png"
    challenge_type = gic.invoke(screenshot_path, model="gemini-2.0-flash")
    assert isinstance(challenge_type, ChallengeTypeEnum)
    assert challenge_type == ChallengeTypeEnum.IMAGE_DRAG_SINGLE


def test_challenge_classifier_image_label_area_select():
    screenshot_path = "challenge_view/image_label_area_select/multi_1.png"
    challenge_type = gic.invoke(screenshot_path, model="gemini-2.0-flash")
    assert isinstance(challenge_type, ChallengeTypeEnum)
    assert challenge_type == ChallengeTypeEnum.IMAGE_LABEL_MULTI_SELECT
