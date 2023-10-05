# -*- coding: utf-8 -*-
# Time       : 2022/2/15 17:43
# Author     : QIN2DIM
# GitHub     : https://github.com/QIN2DIM
# Description:
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.parse import urlparse

from hcaptcha_challenger.components.image_classifier import Classifier as BinaryClassifier
from hcaptcha_challenger.components.image_classifier import LocalBinaryClassifier
from hcaptcha_challenger.components.image_label_area_select import AreaSelector
from hcaptcha_challenger.components.prompt_handler import (
    label_cleaning,
    diagnose_task,
    split_prompt_message,
    prompt2task,
)
from hcaptcha_challenger.onnx.modelhub import DEFAULT_KEYPOINT_MODEL
from hcaptcha_challenger.onnx.modelhub import ModelHub
from hcaptcha_challenger.onnx.resnet import ResNetControl
from hcaptcha_challenger.onnx.yolo import YOLOv8
from hcaptcha_challenger.utils import init_log

__all__ = [
    "BinaryClassifier",
    "LocalBinaryClassifier",
    "AreaSelector",
    "label_cleaning",
    "diagnose_task",
    "split_prompt_message",
    "prompt2task",
    "ModelHub",
    "DEFAULT_KEYPOINT_MODEL",
    "ResNetControl",
    "YOLOv8",
    "install",
]


@dataclass
class Project:
    at_dir = Path(__file__).parent
    logs = at_dir.joinpath("logs")


project = Project()

init_log(
    runtime=project.logs.joinpath("runtime.log"),
    error=project.logs.joinpath("error.log"),
    serialize=project.logs.joinpath("serialize.log"),
)


def install(
    upgrade: bool | None = False,
    username: str = "QIN2DIM",
    lang: str = "en",
    flush_yolo: bool | Iterable[str] = False,
):
    modelhub = ModelHub.from_github_repo(username=username, lang=lang)
    modelhub.pull_objects(upgrade=upgrade)
    modelhub.assets.flush_runtime_assets(upgrade=upgrade)

    if flush_yolo is not None:
        from hcaptcha_challenger.onnx.modelhub import DEFAULT_KEYPOINT_MODEL

        modelhub.parse_objects()

        if isinstance(flush_yolo, bool) and flush_yolo:
            flush_yolo = [DEFAULT_KEYPOINT_MODEL]
        if isinstance(flush_yolo, Iterable):
            pending_models = []
            for model_name in flush_yolo:
                if model_name in modelhub.ashes_of_war:
                    modelhub.pull_model(model_name)
                    pending_models.append(model_name)
            return pending_models


def set_reverse_proxy(https_cdn: str):
    parser = urlparse(https_cdn)
    if parser.netloc and parser.scheme.startswith("https"):
        ModelHub.CDN_PREFIX = https_cdn
