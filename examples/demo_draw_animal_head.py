# -*- coding: utf-8 -*-
# Time       : 2023/9/27 15:28
# Author     : QIN2DIM
# GitHub     : https://github.com/QIN2DIM
# Description:
import os
import sys
from pathlib import Path

import hcaptcha_challenger as solver
from hcaptcha_challenger.components.yolo_mocker import CcYOLO

solver.install(upgrade=True)


def spawn(model_name, images_dir, output_dir):
    ccy = CcYOLO(model_name, images_dir, output_dir)
    ccy.spawn()

    return ccy.output_dir


def demo():
    groups = [
        ("please click on the head of the animal", "default"),
        ("please click on the head of the animal", "polarbearonthesnow"),
        ("please click on the head of the animal", "zebraonthedesert"),
    ]
    assets_dir = Path(__file__).parent.parent.joinpath("assets", "image_label_area_select")
    model_name = "head_of_the_animal_2310_yolov8s.onnx"

    for group in groups:
        images_dir = assets_dir.joinpath(*group).absolute()
        output_dir = Path(__file__).parent.joinpath("figs-detect-out")
        output_dir = spawn(model_name, images_dir, output_dir)

        if "win32" in sys.platform:
            os.startfile(output_dir)


if __name__ == "__main__":
    demo()
