# -*- coding: utf-8 -*-
# Time       : 2022/1/16 0:25
# Author     : QIN2DIM
# Github     : https://github.com/QIN2DIM
# Description:
from typing import Optional

from apis.scaffold import install, challenge


class Scaffold:
    """系统脚手架 顶级接口指令"""

    @staticmethod
    def install(model:Optional[str]="yolov5s6"):
        """下载运行依赖"""
        install.run(model=model)

    @staticmethod
    def demo(silence: Optional[bool] = False, model: Optional[str] = "yolov5s6"):
        """正面硬刚人机挑战"""
        challenge.demo(silence=silence, onnx_prefix=model)
