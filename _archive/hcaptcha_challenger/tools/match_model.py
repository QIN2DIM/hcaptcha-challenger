# -*- coding: utf-8 -*-
# Time       : 2023/11/18 22:38
# Author     : QIN2DIM
# GitHub     : https://github.com/QIN2DIM
# Description:
from __future__ import annotations

from typing import Literal

from hcaptcha_challenger.onnx.modelhub import ModelHub
from hcaptcha_challenger.onnx.resnet import ResNetControl
from hcaptcha_challenger.onnx.yolo import YOLOv8


def match_model(
    label: str, ash: str, modelhub: ModelHub, select: Literal["yolo", "resnet"] = None
) -> ResNetControl | YOLOv8:
    """match solution after `tactical_retreat`"""
    focus_label = modelhub.label_alias.get(label, "")

    # Match YOLOv8 model
    if not focus_label or select == "yolo":
        focus_name, classes = modelhub.apply_ash_of_war(ash=ash)
        session = modelhub.match_net(focus_name=focus_name)
        detector = YOLOv8.from_pluggable_model(session, classes)
        return detector

    # Match ResNet model
    focus_name = focus_label
    if not focus_name.endswith(".onnx"):
        focus_name = f"{focus_name}.onnx"
    net = modelhub.match_net(focus_name=focus_name)
    control = ResNetControl.from_pluggable_model(net)
    return control
