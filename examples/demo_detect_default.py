# -*- coding: utf-8 -*-
# Time       : 2023/9/17 23:46
# Author     : QIN2DIM
# GitHub     : https://github.com/QIN2DIM
# Description:
from pathlib import Path

from loguru import logger

from hcaptcha_challenger import YOLOv8
from hcaptcha_challenger.onnx.modelhub import ModelHub

modelhub = ModelHub.from_github_repo()
modelhub.parse_objects()

image_path = Path("zebora-head.png")

series = "head"


def lookup_yolo():
    for focus_name, covered_class in modelhub.ashes_of_war.items():
        if "animalhead" not in focus_name:
            continue
        session = modelhub.match_net(focus_name=focus_name)
        detector = YOLOv8.from_pluggable_model(session, covered_class)

        res = detector(Path(image_path), shape_type="point")
        if not res:
            continue

        logger.debug("match model", yolo=focus_name, res=res)
        # return res


if __name__ == "__main__":
    lookup_yolo()
