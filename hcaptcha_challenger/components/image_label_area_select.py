# -*- coding: utf-8 -*-
# Time       : 2023/8/29 14:13
# Author     : QIN2DIM
# GitHub     : https://github.com/QIN2DIM
# Description:
from __future__ import annotations

import re
from pathlib import Path
from typing import List, Literal

from loguru import logger

from hcaptcha_challenger.components.prompt_handler import split_prompt_message, label_cleaning
from hcaptcha_challenger.onnx.modelhub import ModelHub
from hcaptcha_challenger.onnx.yolo import apply_ash_of_war, YOLOv8


class AreaSelector:
    def __init__(self):
        self.modelhub = ModelHub.from_github_repo()
        self.modelhub.parse_objects()

    def execute(
        self,
        prompt: str,
        images: List[Path | bytes],
        shape_type: Literal["point", "bounding_box"] = "point",
    ) -> List[bool | None]:
        """

        :param prompt:
        :param images:
        :param shape_type:
        :return:
        """
        response = []

        lang = "zh" if re.compile("[\u4e00-\u9fa5]+").search(prompt) else "en"
        _label = split_prompt_message(prompt, lang=lang)
        label = label_cleaning(_label)

        focus_name, yolo_classes = apply_ash_of_war(ash=label)
        session = self.modelhub.match_net(focus_name=focus_name)
        detector = YOLOv8.from_pluggable_model(session, focus_name)

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
