# -*- coding: utf-8 -*-
# Time       : 2022/1/16 0:25
# Author     : QIN2DIM
# Github     : https://github.com/QIN2DIM
# Description:
from typing import Optional

from apis.scaffold import install, challenge
from services.settings import HCAPTCHA_DEMO_SITES


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
        challenge.runner(HCAPTCHA_DEMO_SITES[0], silence=silence, onnx_prefix=model)

    @staticmethod
    def demo_v2(silence: Optional[bool] = False):
        """Processing hCAPTCHA challenges using Image-Segmentation"""
        # label: vertical river
        challenge.runner(HCAPTCHA_DEMO_SITES[1], silence=silence)

    @staticmethod
    def demo_v3(silence: Optional[bool] = False):
        """Processing hCAPTCHA challenges using Image-Segmentation"""
        # label: airplane in the sky flying left
        challenge.runner(HCAPTCHA_DEMO_SITES[2], silence=silence)
