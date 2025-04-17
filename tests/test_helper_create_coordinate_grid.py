import itertools
import logging
import os
from concurrent.futures import as_completed, ProcessPoolExecutor
from pathlib import Path
from typing import Dict

import numpy as np
from PIL import Image

from hcaptcha_challenger.helper.create_coordinate_grid import create_coordinate_grid

BASE_PATH = Path("challenge_view")
DATASET_IMAGE_DRAG_DROP = BASE_PATH / "image_drag_drop"
DATASET_IMAGE_LABEL_AREA_SELECT = BASE_PATH / "image_label_area_select"

DATASET_IMAGE_DRAG_DROP.mkdir(parents=True, exist_ok=True)
DATASET_IMAGE_LABEL_AREA_SELECT.mkdir(parents=True, exist_ok=True)

DEFAULT_BBOX: Dict[str, int] = {"x": 0, "y": 0, "width": 501, "height": 431}

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

PREFIX_ = "coordinate_grid"
PIL_AVAILABLE = True


def process_and_save_grid(challenge_screenshot: Path, bbox: Dict[str, int]):
    """
    Create a coordinate grid for a single image and save the results.

    Args:
        challenge_screenshot: Enter the path to the image file.
        bbox: Bounding box for create_coordinate_grid.
    """
    try:
        grid_divisions_path = challenge_screenshot.parent / f'{PREFIX_}_{challenge_screenshot.name}'

        # Call the core processing function (assuming it returns an image data suitable for saving, such as a NumPy array)
        result_data = create_coordinate_grid(
            challenge_screenshot,
            bbox,
            adaptive_contrast=False,
            x_line_space_num=11,
            y_line_space_num=20,
            grayscale=True,
        )

        if PIL_AVAILABLE and isinstance(result_data, np.ndarray):
            # Make sure the data type is suitable for Pillow save (usually uint8)
            if result_data.dtype != np.uint8:
                # If it is a floating point number (0-1 range), convert to uint8
                if result_data.max() <= 1.0 and result_data.min() >= 0.0:
                    result_data = (result_data * 255).astype(np.uint8)
                else:
                    # Other situations may require more complex normalization or type conversion
                    logging.warning(
                        f"Result data for {challenge_screenshot.name} has dtype {result_data.dtype}. Attempting direct conversion to uint8."
                    )
                    # May be inaccurate, depending on the original range
                    result_data = result_data.astype(np.uint8)

            # Save NumPy arrays using Pillow
            img = Image.fromarray(result_data)
            img.save(grid_divisions_path)
            logging.info(f"Saved grid image to: {grid_divisions_path} using Pillow")
        else:
            logging.error(
                f"Processing failed for {challenge_screenshot.name}: create_coordinate_grid did not return a NumPy array."
            )

    except FileNotFoundError:
        logging.error(f"Input image not found: {challenge_screenshot}")
    except Exception as e:
        logging.error(f"Failed to process {challenge_screenshot.name}: {e}", exc_info=True)


def test_create_coordinate_grid_parallel(max_workers: int = None):
    """
    Find all the challenge screenshots, process and save the coordinate grid image in parallel.

    Args:
        max_workers: The maximum number of processes to be used. The default is the number of CPU cores of the system.
    """
    if not PIL_AVAILABLE:
        logging.warning(
            "Consider installing Pillow (`pip install Pillow`) for potentially better performance and reduced dependencies."
        )

    if max_workers is None:
        max_workers = os.cpu_count()
        logging.info(f"Using default max_workers: {max_workers}")

    # Get a list of all image paths (parallel processing usually requires collecting all tasks first)
    all_image_paths = list(
        itertools.chain(
            DATASET_IMAGE_DRAG_DROP.rglob("*.png"), DATASET_IMAGE_LABEL_AREA_SELECT.rglob("*.png")
        )
    )

    valid_image_paths = [
        p for p in all_image_paths if p.is_file() and not p.name.startswith(PREFIX_)
    ]
    logging.info(f"Found {len(valid_image_paths)} image files to process.")

    if not valid_image_paths:
        logging.warning("No valid image files found.")
        return

    processed_count = 0
    # Use ProcessPoolExecutor to parallelize CPU-intensive tasks
    # Note: The process_and_save_grid function must be picked up in a child process
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        futures = {
            executor.submit(process_and_save_grid, img_path, DEFAULT_BBOX): img_path
            for img_path in valid_image_paths
        }

        # Handle completed tasks
        for future in as_completed(futures):
            img_path = futures[future]
            try:
                # If process_and_save_grid has a return value, you can get future.result() here
                future.result()  # Check whether the task throws an exception
                processed_count += 1
                # You can add more detailed success logs here, but there are logs inside process_and_save_grid
            except Exception as e:
                # The exception has been recorded in process_and_save_grid.
                # You can record it again or take other measures here.
                logging.error(f"Error processing {img_path} in parallel worker: {e}")

    logging.info(
        f"Finished parallel processing. Processed {processed_count}/{len(valid_image_paths)} images."
    )


if __name__ == "__main__":
    test_create_coordinate_grid_parallel()
