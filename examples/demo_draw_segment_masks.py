# -*- coding: utf-8 -*-
# Time       : 2023/10/15 23:55
# Author     : QIN2DIM
# GitHub     : https://github.com/QIN2DIM
# Description:
import os
import sys
from pathlib import Path

import cv2
from tqdm import tqdm

import hcaptcha_challenger as solver
from hcaptcha_challenger.onnx.modelhub import ModelHub
from hcaptcha_challenger.onnx.yolo import YOLOv8Seg

solver.install(upgrade=True)

# Initialize model index
modelhub = ModelHub.from_github_repo()
modelhub.parse_objects()


def yolov8_segment(images_dir: Path, output_dir: Path):
    # Load model (automatic download, 51MB)
    model_name = "appears_only_once_2309_yolov8s-seg.onnx"
    classes = modelhub.ashes_of_war.get(model_name)
    session = modelhub.match_net(model_name)
    yoloseg = YOLOv8Seg.from_pluggable_model(session, classes)

    pending_image_paths = [images_dir.joinpath(image_name) for image_name in os.listdir(images_dir)]

    # Initialize progress bar
    total = len(pending_image_paths)
    desc_in = f'"{images_dir.parent.name}/{images_dir.name}"'

    with tqdm(total=total, desc=f"Labeling | {desc_in}") as progress:
        for image_path in pending_image_paths:
            # Find all the circles in the picture
            yoloseg(image_path, shape_type="point")

            # Draw a bounding box and mask region for all circles
            img = cv2.imread(str(image_path))
            combined_img = yoloseg.draw_masks(img, mask_alpha=0.3)
            output_path = output_dir.joinpath(image_path.name)
            cv2.imwrite(str(output_path), combined_img)

            progress.update(1)


def demo(startfile=True):
    # fmt:off
    groups = [
        ("please click the center of a circle where all the shapes are of the same color", "default"),
        ("please click the center of the object that is never repeated", "default"),
        ("please click on the object that appears only once", "default"),
    ]
    # fmt:on
    assets_dir = Path(__file__).parent.parent.joinpath("assets", "image_label_area_select")

    for group in groups:
        images_dir = assets_dir.joinpath(*group)
        output_dir = Path(__file__).parent.joinpath("figs-seg-out", group[0])
        output_dir.mkdir(parents=True, exist_ok=True)
        yolov8_segment(images_dir, output_dir)

        if "win32" in sys.platform and startfile:
            os.startfile(output_dir)
        print(f">> View at {output_dir}")


if __name__ == "__main__":
    demo()
