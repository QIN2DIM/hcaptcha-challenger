import os
from pathlib import Path

import dotenv
from loguru import logger
from matplotlib import pyplot as plt

from hcaptcha_challenger import SpatialBboxReasoner
from hcaptcha_challenger.helper import create_coordinate_grid, FloatRect

dotenv.load_dotenv()
gic = SpatialBboxReasoner(gemini_api_key=os.getenv("GEMINI_API_KEY"), model='gemini-2.5-flash')

CHALLENGE_VIEW_DIR = Path(__file__).parent.joinpath("challenge_view/image_drag_drop")


async def test_gemini_bbox_reasoning():
    challenge_screenshot = CHALLENGE_VIEW_DIR.joinpath("single_5.png")
    grid_divisions_path = challenge_screenshot.parent.joinpath(
        f'coordinate_grid_{challenge_screenshot.name}'
    )
    bbox = FloatRect(x=0, y=0, width=501, height=431)

    grid_divisions_image = create_coordinate_grid(challenge_screenshot, bbox)
    plt.imsave(str(grid_divisions_path.resolve()), grid_divisions_image)

    results = await gic(
        challenge_screenshot=challenge_screenshot, grid_divisions=grid_divisions_path
    )
    logger.debug(f'ToolInvokeMessage: {results.log_message}')
