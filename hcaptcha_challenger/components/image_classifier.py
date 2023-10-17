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

        self.response: List[bool | None] = []
        self.prompt: str = ""
        self.label: str = ""
        self.model_name: str = ""

    def _parse_label(self, prompt: str):
        self.prompt = prompt
        lang = "zh" if re.compile("[\u4e00-\u9fa5]+").search(prompt) else "en"
        _label = label_cleaning(prompt)
        _label = split_prompt_message(_label, lang=lang)
        self.label = _label

    def rank_models(
        self, nested_models: List[str], example_paths: List[Path | bytes]
    ) -> ResNetControl | None:
        # {{< Rank ResNet Models >}}
        rank_ladder = []
        for example_path in example_paths:
            if isinstance(example_path, bytes):
                img_stream = example_path
            elif isinstance(example_path, Path):
                img_stream = example_path.read_bytes()
            else:
                continue

            for model_name in nested_models:
                net = self.modelhub.match_net(focus_name=model_name)
                control = ResNetControl.from_pluggable_model(net)
                result_, proba = control.execute(img_stream, proba=True)
                if result_:
                    rank_ladder.append([control, model_name, proba])
                    if proba[0] > 0.87:
                        break

        # {{< Catch-all Rules >}}
        if rank_ladder:
            alts = sorted(rank_ladder, key=lambda x: x[-1][0], reverse=True)
            best_model, model_name = alts[0][0], alts[0][1]
            self.model_name = model_name
            return best_model

    def inference(self, images: List[Path | bytes], control: ResNetControl):
        for image in images:
            try:
                if isinstance(image, Path):
                    if not image.exists():
                        self.response.append(None)
                        continue
                    image = image.read_bytes()
                if isinstance(image, bytes):
                    result = control.execute(image)
                    self.response.append(result)
                else:
                    self.response.append(None)
            except Exception as err:
                logger.debug(str(err), prompt=self.prompt)
                self.response.append(None)

    def execute(
        self,
        prompt: str,
        images: List[Path | bytes],
        example_paths: List[Path | bytes] | None = None,
    ) -> List[bool | None]:
        self.response = []

        self._parse_label(prompt)

        # Match: model ranker
        if nested_models := self.modelhub.nested_categories.get(self.label, []):
            if model := self.rank_models(nested_models, example_paths):
                self.inference(images, model)
        # Match: common binary classification
        elif focus_label := self.modelhub.label_alias.get(self.label):
            self.model_name = (
                focus_label if focus_label.endswith(".onnx") else f"{focus_label}.onnx"
            )
            net = self.modelhub.match_net(self.model_name)
            control = ResNetControl.from_pluggable_model(net)
            self.inference(images, control)
        # Match: Unknown cases
        else:
            logger.debug("Types of challenges not yet scheduled", label=self.label, prompt=prompt)

        return self.response


class LocalBinaryClassifier:
    def __init__(self, model_path: Path):
        if not isinstance(model_path, Path) or not model_path.exists():
            raise FileNotFoundError
        self.model_path = model_path
        net = cv2.dnn.readNetFromONNX(str(model_path))
        self.model = ResNetControl.from_pluggable_model(net)

    def parse_once(self, image: bytes) -> bool | None:
        with suppress(Exception):
            return self.model.execute(image)
