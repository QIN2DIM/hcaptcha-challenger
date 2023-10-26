# -*- coding: utf-8 -*-
# Time       : 2023/8/29 14:13
# Author     : QIN2DIM
# GitHub     : https://github.com/QIN2DIM
# Description:
from __future__ import annotations

from pathlib import Path
from typing import List, Literal

from loguru import logger

from hcaptcha_challenger.components.prompt_handler import handle
from hcaptcha_challenger.onnx.modelhub import ModelHub
from hcaptcha_challenger.onnx.yolo import YOLOv8


class AreaSelector:
    def __init__(self):
        self.modelhub = ModelHub.from_github_repo()
        self.modelhub.parse_objects()

    def execute(
        self,
        prompt: str,
        images: List[Path | bytes],
        shape_type: Literal["point", "bounding_box"] = "point",
        *,
        answer_key: str = "",
    ) -> List[tuple | None]:
        """
        answer_keys = list(self.qr.requester_restricted_answer_set.keys())
        ak = answer_keys[0] if len(answer_keys) > 0 else ""
        ash = f"{self._label} {ak}"

        :param answer_key:
        :param prompt:
        :param images:
        :param shape_type:
        :return:

        IF shape_type == point
            Element --> (class_name, (center_x, center_y), score)
            Response --> List[Element | None]
        ELIF shape_type == bounding box
            Element -->  (class_name, (x1, y1), (x2, y2), score)
            Response --> List[Element | None]
        """
        response = []

        ash = f"{handle(prompt)} {answer_key}"

        focus_name, classes = self.modelhub.apply_ash_of_war(ash=ash)
        session = self.modelhub.match_net(focus_name=focus_name)
        if not session:
            logger.error(
                f"ModelNotFound, please upgrade assets and flush yolo model", focus_name=focus_name
            )
            return response
        detector = YOLOv8.from_pluggable_model(session, classes)

        for image in images:
            try:
                if isinstance(image, Path):
                    if not image.exists():
                        response.append(None)
                        continue
                    image = image.read_bytes()
                if isinstance(image, bytes):
                    result = detector(image=image, shape_type=shape_type)
                    response.append(result)
                else:
                    response.append(None)
            except Exception as err:
                logger.debug(str(err), prompt=prompt)
                response.append(None)

        return response
