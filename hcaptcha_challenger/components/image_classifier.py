# -*- coding: utf-8 -*-
# Time       : 2023/8/19 17:53
# Author     : QIN2DIM
# GitHub     : https://github.com/QIN2DIM
# Description:
from __future__ import annotations

import re
from contextlib import suppress
from pathlib import Path
from typing import List

import cv2
from loguru import logger

from hcaptcha_challenger.components.prompt_handler import label_cleaning, split_prompt_message
from hcaptcha_challenger.onnx.modelhub import ModelHub
from hcaptcha_challenger.onnx.resnet import ResNetControl


class Classifier:
    def __init__(self):
        self.modelhub = ModelHub.from_github_repo()
        self.modelhub.parse_objects()

    def execute(self, prompt: str, images: List[Path | bytes]) -> List[bool | None]:
        response = []

        lang = "zh" if re.compile("[\u4e00-\u9fa5]+").search(prompt) else "en"
        _label = split_prompt_message(prompt, lang=lang)
        label = label_cleaning(_label)

        focus_label = self.modelhub.label_alias.get(label)
        if not focus_label:
            logger.debug("Types of challenges not yet scheduled", label=label, prompt=prompt)
            return response

        focus_name = focus_label if focus_label.endswith(".onnx") else f"{focus_label}.onnx"
        net = self.modelhub.match_net(focus_name)
        control = ResNetControl.from_pluggable_model(net)

        for image in images:
            try:
                if isinstance(image, Path):
                    if not image.exists():
                        response.append(None)
                        continue
                    image = image.read_bytes()
                if isinstance(image, bytes):
                    result = control.binary_classify(image)
                    response.append(result)
                else:
                    response.append(None)
            except Exception as err:
                logger.debug(str(err), label=focus_label, prompt=prompt)
                response.append(None)

        return response


class LocalBinaryClassifier:
    def __init__(self, model_path: Path):
        if not isinstance(model_path, Path) or not model_path.exists():
            raise FileNotFoundError
        self.model_path = model_path
        net = cv2.dnn.readNetFromONNX(str(model_path))
        self.model = ResNetControl.from_pluggable_model(net)

    def parse_once(self, image: bytes) -> bool | None:
        with suppress(Exception):
            return self.model.binary_classify(image)
