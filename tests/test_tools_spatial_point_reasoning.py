import os
from pathlib import Path

import dotenv
from loguru import logger
from matplotlib import pyplot as plt

from hcaptcha_challenger import SpatialPointReasoner
from hcaptcha_challenger.helper import create_coordinate_grid, FloatRect
from hcaptcha_challenger.helper.visualize_attention_points import show_answer_points

dotenv.load_dotenv()
gic = SpatialPointReasoner(gemini_api_key=os.getenv("GEMINI_API_KEY"), model="gemini-2.5-flash")

CHALLENGE_VIEW_DIR = Path(__file__).parent.joinpath("challenge_view/image_label_area_select")
SHOW_ANSWER_DIR = Path(__file__).parent.joinpath("show_answer/image_label_area_select")


async def test_gemini_point_reasoning():
    challenge_screenshot = CHALLENGE_VIEW_DIR.joinpath("single_5.png")
    grid_divisions_path = challenge_screenshot.parent.joinpath(
        f'coordinate_grid_{challenge_screenshot.name}'
    )
    bbox = FloatRect(x=0, y=0, width=501, height=431)

    grid_divisions_image = create_coordinate_grid(challenge_screenshot, bbox)
    plt.imsave(str(grid_divisions_path.resolve()), grid_divisions_image)

    results = await gic.invoke_async(
        challenge_screenshot=challenge_screenshot, grid_divisions=grid_divisions_path
    )
    logger.debug(f'ToolInvokeMessage: {results.log_message}')

    # Visualize the answer on the actual image
    result = show_answer_points(
        challenge_screenshot,
        results,
        bbox,
        show_plot=True,
        path_color='blue',
        arrow_width=3,
        alpha=0.7,
    )

    SHOW_ANSWER_DIR.mkdir(parents=True, exist_ok=True)
    save_path = SHOW_ANSWER_DIR.joinpath(challenge_screenshot.name)
    plt.imsave(str(save_path), result)
    logger.info(f"Saved answer visualization to {save_path}")
