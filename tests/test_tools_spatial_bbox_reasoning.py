import os
from pathlib import Path

import dotenv
from loguru import logger
from matplotlib import pyplot as plt

from hcaptcha_challenger.helper import create_coordinate_grid, FloatRect
from hcaptcha_challenger.tools import SpatialBboxReasoner

dotenv.load_dotenv()
gic = SpatialBboxReasoner(gemini_api_key=os.getenv("GEMINI_API_KEY"))


def test_gemini_bbox_reasoning():
    challenge_screenshot = Path("challenge_view/image_drag_drop/single_5.png")
    grid_divisions_path = challenge_screenshot.parent.joinpath(
        f'coordinate_grid_{challenge_screenshot.name}'
    )
    bbox = FloatRect(x=0, y=0, width=501, height=431)

    grid_divisions_image = create_coordinate_grid(challenge_screenshot, bbox)
    plt.imsave(str(grid_divisions_path.resolve()), grid_divisions_image)

    results = gic.invoke(
        challenge_screenshot=challenge_screenshot,
        grid_divisions=grid_divisions_path,
        model="gemini-2.5-flash-preview-04-17",
    )
    logger.debug(f'ToolInvokeMessage: {results.log_message}')
