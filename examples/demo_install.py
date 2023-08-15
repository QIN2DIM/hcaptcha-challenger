# -*- coding: utf-8 -*-
# Time       : 2022/1/20 16:16
# Author     : QIN2DIM
# Github     : https://github.com/QIN2DIM
# Description:
from __future__ import annotations

import hcaptcha_challenger as solver


def do(yolo_onnx_prefix: str | None = None, upgrade: bool | None = False):
    """Download the dependencies required to run the project."""
    onnx_prefix = yolo_onnx_prefix or solver.Prefix.YOLOv6n
    solver.install(onnx_prefix=onnx_prefix, upgrade=upgrade)


if __name__ == "__main__":
    do(upgrade=True)
