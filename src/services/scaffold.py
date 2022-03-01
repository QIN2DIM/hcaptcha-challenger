# -*- coding: utf-8 -*-
# Time       : 2022/1/16 0:25
# Author     : QIN2DIM
# Github     : https://github.com/QIN2DIM
# Description:
from typing import Optional

from apis.scaffold import install, challenge


class Scaffold:
    """System scaffolding Top-level interface commands"""

    @staticmethod
    def install(model: Optional[str] = None):
        """Download Project Dependencies"""
        install.run(model=model)

    @staticmethod
    def test():
        """Test the Challenger drive for fitment"""
        challenge.test()

    @staticmethod
    def demo(silence: Optional[bool] = False, model: Optional[str] = None):
        """Dueling with hCAPTCHA challenge using YOLOv5"""
        challenge.demo(silence=silence, onnx_prefix=model)

    @staticmethod
    def demo_v2(silence: Optional[bool] = False, model: Optional[str] = None):
        """Processing hCAPTCHA challenges using Image-Segmentation"""
        challenge.demo_v2(silence=silence,onnx_prefix=model)
