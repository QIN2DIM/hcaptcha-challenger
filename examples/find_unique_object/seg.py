import os
import sys
from pathlib import Path
from typing import Tuple, Callable, Literal

import cv2
from tqdm import tqdm

import hcaptcha_challenger as solver
from hcaptcha_challenger.components.cv_toolkit.appears_only_once import (
    limited_radius,
    annotate_objects,
    find_unique_object,
    find_unique_color,
)
from hcaptcha_challenger.onnx.modelhub import ModelHub
from hcaptcha_challenger.onnx.yolo import YOLOv8Seg

solver.install(upgrade=True)


def in2trent(figs_in: Literal["figs1", "figs2", "figs3"]) -> Tuple[Path, Callable]:
    """
        # figs1 - screenshot of challenge-view canvas
        # figs2 - challenge images --> Objects based
        # figs3 - challenge images --> ColorBlock based
    :param figs_in:
    :return:
    """
    i2t_mapping = {
        "figs1": find_unique_object,
        "figs2": find_unique_object,
        "figs3": find_unique_color,
    }
    return Path(__file__).parent.joinpath(figs_in), i2t_mapping[figs_in]


# !! IMPORT !! select input dataset
figs_dir, trident = in2trent(figs_in="figs1")
figs_out_dir = Path(f"{figs_dir.name}-seg-out")
figs_out_dir.mkdir(exist_ok=True)

# Initialize model index
modelhub = ModelHub.from_github_repo()
modelhub.parse_objects()


def draw_unique_object(results, image_path: str):
    def search():
        # Hough circle detection as a bottoming scheme
        img, circles = annotate_objects(image_path)

        # Prioritize the results of seg model cutting
        # IF NOT results - Use Hough's results
        if results:
            # circle: [center_x, center_y, r]
            circles = [
                [int(result[1][0]), int(result[1][1]), limited_radius(img)] for result in results
            ]

        # Find the circle field of the `appears only once`
        if circles:
            if result := trident(img, circles):
                x, y, _ = result
                # Return the `Position` DictType[str, int]
                return img, {"x": int(x), "y": int(y)}

        # `trident() method` returned untrusted results
        return img, {}

    # IF position is reliable - Mark the center of a unique circle
    image, position = search()
    # print(position)

    # If you are writing a playwright or selenium program,
    # you should click canvas according to this coordinate.
    if position:
        combined_img = cv2.circle(
            image, (position["x"], position["y"]), limited_radius(image), (255, 0, 0), 2
        )
        return combined_img
    return image


def run():
    # Load model (automatic download, 51MB)
    model_name = "appears_only_once_2309_yolov8s-seg.onnx"
    classes = modelhub.ashes_of_war.get(model_name)
    session = modelhub.match_net(model_name)
    yoloseg = YOLOv8Seg.from_pluggable_model(session, classes)

    # Read data set
    pending_image_paths = [
        figs_dir.joinpath(image_name) for image_name in os.listdir(figs_dir)[:100]
    ]

    # Initialize progress bar
    total = len(pending_image_paths)
    desc_in = f'"{figs_dir.parent.name}/{figs_dir.name}"'

    with tqdm(total=total, desc=f"Labeling | {desc_in}") as progress:
        for image_path in pending_image_paths:
            # Find all the circles in the picture
            results = yoloseg(image_path)

            # Find the unique circle and draw the center of the circle
            combined_img = draw_unique_object(results, str(image_path))
            # Draw a bounding box and mask region for all circles
            combined_img = yoloseg.draw_masks(combined_img, mask_alpha=0.1)

            # Preserve pictures with traces of drawing
            output_path = figs_out_dir.joinpath(image_path.name)
            cv2.imwrite(str(output_path), combined_img)

            progress.update(1)

    if "win32" in sys.platform:
        os.startfile(figs_out_dir)


# pip install -U hcaptcha_challenger
if __name__ == "__main__":
    run()
