# -*- coding: utf-8 -*-
# Time       : 2022/1/16 0:25
# Author     : QIN2DIM
# Github     : https://github.com/QIN2DIM
# Description:
from typing import Optional

from apis.scaffold import install, challenge, app, motion
from services.settings import HCAPTCHA_DEMO_SITES, _SITE_KEYS, HCAPTCHA_DEMO_API


class Scaffold:
    """System scaffolding Top-level interface commands"""

    challenge_language = "zh"

    def __init__(self, lang: Optional[str] = None):
        if lang is not None:
            Scaffold.challenge_language = lang

    @staticmethod
    def install(model: Optional[str] = None, upgrade: Optional[bool] = True):
        """Download Project Dependencies and upgrade all pluggable ONNX model"""
        install.run(model=model, upgrade=upgrade)

    @staticmethod
    def test():
        """Test the Challenger drive for fitment"""
        challenge.test()

    @staticmethod
    def demo(
        silence: Optional[bool] = False,
        model: Optional[str] = None,
        target: Optional[str] = None,
        sitekey: Optional[str] = None,
        screenshot: Optional[bool] = False,
    ):
        """
        Dueling with hCAPTCHA challenge using YOLOv5.

        Usage: python main.py demo
        ___________________________________________________
        or: python main.py demo --model=yolov5n6     |
        or: python main.py demo --target=discord     |
        or: python main.py demo --lang=en            |
        or: python main.py demo --sitekey=[UUID]     |
        ---------------------------------------------------
        :param screenshot: save screenshot of the challenge result to ./database/temp_cache/captcha_screenshot/
          FILENAME: ChallengeLabelName.png
        :param sitekey: customize the challenge theme via sitekey
        :param silence: Default False. Whether to silence the browser window.
        :param model: Default "yolov5s6". within [yolov5n6 yolov5s6 yolov5m6 yolov6n yolov6t yolov6s]
        :param target: Default None. Designate `Challenge Source`. See the global value SITE_KEYS.
        :return:
        """

        # Generate challenge topics
        if _SITE_KEYS.get(target):
            sample_site = HCAPTCHA_DEMO_API.format(_SITE_KEYS[target])
        else:
            sample_site = HCAPTCHA_DEMO_SITES[0]
        if sitekey is not None:
            sample_site = HCAPTCHA_DEMO_API.format(sitekey.strip())

        # Pre-check the Pluggable ONNX model
        # install.refresh_pluggable_onnx_model(upgrade=True)

        challenge.runner(
            sample_site,
            lang=Scaffold.challenge_language,
            silence=silence,
            onnx_prefix=model,
            screenshot=screenshot,
        )

    @staticmethod
    def tracker():
        app.run(debug=True, access_log=False)

    @staticmethod
    def motion():
        motion.train_motion("http://127.0.0.1:8000")
