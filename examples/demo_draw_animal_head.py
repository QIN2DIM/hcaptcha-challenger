# -*- coding: utf-8 -*-
# Time       : 2023/9/27 15:28
# Author     : QIN2DIM
# GitHub     : https://github.com/QIN2DIM
# Description:
import os
import sys
from pathlib import Path

from hcaptcha_challenger.components.yolo_mocker import CcYOLO


def run():
    this_dir = Path(__file__).parent
    output_dir = this_dir.joinpath("yolo_mocker")

    model_name_to_images = {
        "head_of_the_animal_2310_yolov8s.onnx": [
            "image_label_area_select/please click on the head of the animal/default",
            "image_label_area_select/please click on the head of the animal/polarbearonthesnow",
            "image_label_area_select/please click on the head of the animal/zebraonthedesert"
        ]
    }

    for model_name, images_dir_group in model_name_to_images.items():
        for images_dir in images_dir_group:
            images_dir = this_dir.joinpath(images_dir).absolute()

            ccy = CcYOLO(model_name, images_dir, output_dir)
            ccy.spawn()

            if "win32" in sys.platform:
                os.startfile(ccy.output_dir)


if __name__ == "__main__":
    run()
