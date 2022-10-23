# -*- coding: utf-8 -*-
# Time       : 2022/1/20 16:16
# Author     : QIN2DIM
# Github     : https://github.com/QIN2DIM
# Description:
import typing

import hcaptcha_challenger as solver


def do(yolo_onnx_prefix: typing.Optional[str] = None, upgrade: typing.Optional[bool] = False):
    """下载项目运行所需的各项依赖"""
    onnx_prefix = yolo_onnx_prefix or solver.Prefix.YOLOv6n
    solver.install(onnx_prefix=onnx_prefix, upgrade=upgrade)
