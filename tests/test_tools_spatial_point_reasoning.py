import os
from pathlib import Path

import dotenv
from loguru import logger
from matplotlib import pyplot as plt

from hcaptcha_challenger.helper import create_coordinate_grid
from hcaptcha_challenger.tools import SpatialPointReasoner

dotenv.load_dotenv()
gic = SpatialPointReasoner(gemini_api_key=os.getenv("GEMINI_API_KEY"))


def test_gemini_image_classifier():
    challenge_screenshot = Path("challenge_view/image_label_area_select/multi_1.png")
    grid_divisions_path = challenge_screenshot.parent.joinpath(
        f'grid_divisions_{challenge_screenshot.name}'
    )
    bbox = {"x": 0, "y": 0, "width": 769, "height": 793}

    grid_divisions_image = create_coordinate_grid(challenge_screenshot, bbox)
    plt.imsave(str(grid_divisions_path.resolve()), grid_divisions_image)

    results = gic.invoke(
        challenge_screenshot=challenge_screenshot,
        grid_divisions=grid_divisions_path,
        model="gemini-2.5-pro-exp-03-25",
    )
    logger.debug(f'ToolInvokeMessage: {results.log_message}')
