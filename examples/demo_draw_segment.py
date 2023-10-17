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

this_dir = Path(__file__).parent
figs_dir = this_dir.joinpath("figs-seg")
figs_out_dir = this_dir.joinpath("figs-seg-out")


def yolov8_segment():
    figs_out_dir.mkdir(exist_ok=True)

    modelhub = ModelHub.from_github_repo()
    modelhub.parse_objects()

    model_name = "appears_only_once_2309_yolov8s-seg.onnx"
    classes = modelhub.ashes_of_war.get(model_name)
    session = modelhub.match_net(model_name)
    yoloseg = YOLOv8Seg.from_pluggable_model(session, classes)

    pending_image_paths = [figs_dir.joinpath(image_name) for image_name in os.listdir(figs_dir)]

    total = len(pending_image_paths)
    desc_in = f'"{figs_dir.parent.name}/{figs_dir.name}"'
    with tqdm(total=total, desc=f"Labeling | {desc_in}") as progress:
        for image_path in pending_image_paths:
            yoloseg(image_path, shape_type="point")

            img = cv2.imread(str(image_path))
            combined_img = yoloseg.draw_masks(img, mask_alpha=0.3)
            output_path = figs_out_dir.joinpath(image_path.name)
            cv2.imwrite(str(output_path), combined_img)

            progress.update(1)

    if "win32" in sys.platform:
        os.startfile(figs_out_dir)
    print(f">> View at {figs_out_dir}")


if __name__ == "__main__":
    yolov8_segment()
