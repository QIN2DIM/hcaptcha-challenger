# -*- coding: utf-8 -*-
# Time       : 2022/2/15 17:43
# Author     : QIN2DIM
# GitHub     : https://github.com/QIN2DIM
# Description:
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from hcaptcha_challenger.components.image_classifier import Classifier as BinaryClassifier
from hcaptcha_challenger.components.image_label_area_select import AreaSelector
from hcaptcha_challenger.components.prompt_handler import (
    label_cleaning,
    diagnose_task,
    split_prompt_message,
    prompt2task,
)
from hcaptcha_challenger.onnx.modelhub import ModelHub
from hcaptcha_challenger.onnx.resnet import ResNetControl
from hcaptcha_challenger.onnx.yolo import YOLOv8
from hcaptcha_challenger.utils import init_log

__all__ = [
    "BinaryClassifier",
    "AreaSelector",
    "install",
    "YOLOv8",
    "ResNetControl",
    "label_cleaning",
    "diagnose_task",
    "split_prompt_message",
    "prompt2task",
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
    flush_yolo: bool = False,
):
    modelhub = ModelHub.from_github_repo(username=username, lang=lang)
    modelhub.pull_objects(upgrade=upgrade)
    modelhub.assets.flush_runtime_assets(upgrade=upgrade)
    if flush_yolo:
        from hcaptcha_challenger.onnx.modelhub import DEFAULT_KEYPOINT_MODEL

        modelhub.pull_model(focus_name=DEFAULT_KEYPOINT_MODEL)


def set_reverse_proxy(https_cdn: str):
    parser = urlparse(https_cdn)
    if parser.netloc and parser.scheme.startswith("https"):
        ModelHub.CDN_PREFIX = https_cdn
