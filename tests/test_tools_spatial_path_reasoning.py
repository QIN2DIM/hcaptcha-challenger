import asyncio
import os
from pathlib import Path

import dotenv
import pytest
from loguru import logger
from matplotlib import pyplot as plt

from hcaptcha_challenger import SpatialPathReasoner
from hcaptcha_challenger.helper import create_coordinate_grid, FloatRect
from hcaptcha_challenger.helper.visualize_attention_points import show_answer_points

dotenv.load_dotenv()
gic = SpatialPathReasoner(gemini_api_key=os.getenv("GEMINI_API_KEY"), model="gemini-3-pro-preview")

CHALLENGE_VIEW_DIR = Path(__file__).parent.joinpath("challenge_view/image_drag_drop")
SHOW_ANSWER_DIR = Path(__file__).parent.joinpath("show_answer/image_drag_drop")

IS_COLLECT_SELECT_FILE = True


def _collect_image_files(input_dir: Path = CHALLENGE_VIEW_DIR) -> list[Path]:
    if not input_dir.exists():
        return []

    if IS_COLLECT_SELECT_FILE:
        return [
            # input_dir.joinpath("20251119222723757402_0_challenge_view.png"),
            # input_dir.joinpath("20251120024634612368_0_challenge_view.png"),
            # input_dir.joinpath("20251120030025414246_0_challenge_view.png"),
            # input_dir.joinpath("20251120031219718035_0_challenge_view.png"),
            # input_dir.joinpath("img_4f5p9c.png"),
            input_dir.joinpath("img_imjph9.png")
        ]

    return [
        f
        for f in input_dir.iterdir()
        if f.is_file()
        and f.suffix.lower() in ['.png', '.jpg', '.jpeg']
        and not f.name.startswith('coordinate_grid')
    ]


@pytest.mark.parametrize("challenge_screenshot", _collect_image_files())
async def test_gemini_path_reasoning(challenge_screenshot: Path):
    grid_divisions_path = challenge_screenshot.parent.joinpath(
        f'coordinate_grid_{challenge_screenshot.name}'
    )
    bbox = FloatRect(x=0, y=0, width=500, height=430)

    grid_divisions_image = create_coordinate_grid(challenge_screenshot, bbox)
    plt.imsave(str(grid_divisions_path.resolve()), grid_divisions_image)

    results = await gic(
        challenge_screenshot=challenge_screenshot, grid_divisions=grid_divisions_path
    )
    logger.debug(f'ToolInvokeMessage: {results.log_message}')

    result = show_answer_points(
        challenge_screenshot,
        results,
        bbox,
        show_plot=False,
        path_color='blue',
        arrow_width=3,
        alpha=0.7,
    )

    SHOW_ANSWER_DIR.mkdir(parents=True, exist_ok=True)
    save_path = SHOW_ANSWER_DIR.joinpath(challenge_screenshot.name)
    plt.imsave(str(save_path), result)
    logger.info(f"Saved answer visualization to {save_path}")


async def test_gemini_path_reasoning_concurrent():
    """Process all challenge screenshots concurrently using asyncio.gather"""
    challenge_screenshots = _collect_image_files()

    if not challenge_screenshots:
        pytest.skip("No challenge screenshots found")

    async def process_single_image(challenge_screenshot: Path):
        """Process a single challenge screenshot"""
        grid_divisions_path = challenge_screenshot.parent.joinpath(
            f'coordinate_grid_{challenge_screenshot.name}'
        )
        bbox = FloatRect(x=0, y=0, width=500, height=430)

        grid_divisions_image = create_coordinate_grid(challenge_screenshot, bbox)
        plt.imsave(str(grid_divisions_path.resolve()), grid_divisions_image)

        results = await gic(
            challenge_screenshot=challenge_screenshot, grid_divisions=grid_divisions_path
        )
        logger.debug(f'ToolInvokeMessage for {challenge_screenshot.name}: {results.log_message}')

        result = show_answer_points(
            challenge_screenshot,
            results,
            bbox,
            show_plot=False,
            path_color='blue',
            arrow_width=3,
            alpha=0.7,
        )

        SHOW_ANSWER_DIR.mkdir(parents=True, exist_ok=True)
        save_path = SHOW_ANSWER_DIR.joinpath(challenge_screenshot.name)
        plt.imsave(str(save_path), result)
        logger.info(f"Saved answer visualization to {save_path}")

        return challenge_screenshot.name

    # Process all images concurrently
    logger.info(f"Processing {len(challenge_screenshots)} images concurrently...")
    results = await asyncio.gather(*[process_single_image(img) for img in challenge_screenshots])
    logger.success(f"Successfully processed {len(results)} images concurrently: {results}")
