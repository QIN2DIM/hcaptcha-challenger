# -*- coding: utf-8 -*-
# Time       : 2022/1/16 0:25
# Author     : QIN2DIM
# Github     : https://github.com/QIN2DIM
# Description:
from typing import Optional

from apis.scaffold import install, challenge
from services.settings import HCAPTCHA_DEMO_SITES, _SITE_KEYS, _HCAPTCHA_DEMO_API


class Scaffold:
    """System scaffolding Top-level interface commands"""

    challenge_language = "zh"

    def __init__(self, lang: Optional[str] = None):
        if lang is not None:
            Scaffold.challenge_language = lang

    @staticmethod
    def install(model: Optional[str] = None):
        """Download Project Dependencies"""
        install.run(model=model)

    @staticmethod
    def test():
        """Test the Challenger drive for fitment"""
        challenge.test()

    @staticmethod
    def demo(
        silence: Optional[bool] = False,
        model: Optional[str] = None,
        target: Optional[str] = None,
    ):
        """
        Dueling with hCAPTCHA challenge using YOLOv5.

        Usage: python main.py demo
        ___________________________________________________
        or: python main.py demo --model=yolov5n6     |
        or: python main.py demo --target=discord     |
        or: python main.py demo --lang=en            |
        ---------------------------------------------------

        :param silence: Default：False，Whether to silence the browser window.
        :param model: Default: yolov5s6. within [yolov5n6 yolov5s6 yolov5m6]
        :param target: Default: None. Designate `Challenge Source`. See the global value SITE_KEYS.
        :return:
        """
        if _SITE_KEYS.get(target):
            sample_site = _HCAPTCHA_DEMO_API.format(_SITE_KEYS[target])
        else:
            sample_site = HCAPTCHA_DEMO_SITES[0]

        challenge.runner(
            sample_site,
            lang=Scaffold.challenge_language,
            silence=silence,
            onnx_prefix=model,
        )

    @staticmethod
    def demo_v2(silence: Optional[bool] = False):
        """Processing hCAPTCHA challenges using Image-Segmentation"""
        # label: vertical river
        challenge.runner(
            HCAPTCHA_DEMO_SITES[1], lang=Scaffold.challenge_language, silence=silence
        )

    @staticmethod
    def demo_v3(silence: Optional[bool] = False):
        """Processing hCAPTCHA challenges using Image-Segmentation"""
        # label: airplane in the sky flying left
        challenge.runner(
            HCAPTCHA_DEMO_SITES[2], lang=Scaffold.challenge_language, silence=silence
        )
