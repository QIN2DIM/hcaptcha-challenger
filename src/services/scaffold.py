# -*- coding: utf-8 -*-
# Time       : 2022/1/16 0:25
# Author     : QIN2DIM
# Github     : https://github.com/QIN2DIM
# Description:
import typing

from apis.examples import movement, classify
from apis.scaffold import install, challenge, app, motion
from services.settings import config

install.do()


class Scaffold:
    """System scaffolding Top-level interface commands"""

    CHALLENGE_LANGUAGE = "zh"

    def __init__(self, lang: typing.Optional[str] = None):
        if lang is not None:
            Scaffold.CHALLENGE_LANGUAGE = lang

        self.challenge = self.demo
        self.run = self.demo

    @staticmethod
    def install(model: typing.Optional[str] = None, upgrade: typing.Optional[bool] = False):
        """Download Project Dependencies and upgrade all pluggable ONNX model"""
        install.do(yolo_onnx_prefix=model, upgrade=upgrade)

    @staticmethod
    def test():
        """Test the Challenger drive for fitment"""
        challenge.test()

    @staticmethod
    def tracker():
        app.run(debug=True, access_log=False)

    @staticmethod
    def motion():
        motion.train_motion("http://127.0.0.1:8000")

    @staticmethod
    def demo(
        silence: typing.Optional[bool] = False,
        model: typing.Optional[str] = None,
        target: typing.Optional[str] = None,
        sitekey: typing.Optional[str] = None,
        screenshot: typing.Optional[bool] = False,
        repeat: typing.Optional[int] = 5,
    ):
        """
        Dueling with hCAPTCHA challenge using YOLOv5.

        Usage: python main.py demo [flags]
        ——————————————————————————————————————————————————————————————————
        or: python main.py demo --model=yolov5n6
        or: python main.py demo --target=discord
        or: python main.py demo --lang=en
        or: python main.py demo --sitekey=[UUID]
        ——————————————————————————————————————————————————————————————————

        :param repeat: Default 5. Number of times to repeat the presentation.
        :param screenshot: Default False. Save screenshot of the challenge result.
            PATH: database/temp_cache/captcha_screenshot/
            FILENAME: ChallengeLabelName.png
        :param sitekey: Default None. customize the challenge theme via sitekey
        :param silence: Default False. Whether to startup headless browser.
        :param model: Default "yolov5s6". Select the YOLO model to handle specific challenges.
            within [yolov5n6 yolov5s6 yolov5m6 yolov6n yolov6t yolov6s]
        :param target: Default None. Designate `Challenge Source`. See the global value SITE_KEYS.
        :return:
        """
        # Generate challenge topics
        if config.SITE_KEYS.get(target):
            sample_site = config.HCAPTCHA_DEMO_API.format(config.SITE_KEYS[target])
        else:
            sample_site = config.HCAPTCHA_DEMO_SITES[0]
        if sitekey is not None:
            sample_site = config.HCAPTCHA_DEMO_API.format(sitekey.strip())

        challenge.runner(
            sample_site,
            lang=Scaffold.CHALLENGE_LANGUAGE,
            silence=silence,
            onnx_prefix=model,
            screenshot=screenshot,
            repeat=repeat,
        )

    @staticmethod
    def demo_bytedance():
        """
        signup hcaptcha dashboard

        Usage: python main.py demo-bytedance
        :return:
        """
        movement()

    @staticmethod
    def demo_classify():
        """
        BinaryClassification Task

        Usage: python main.py demo-classify
        :return:
        """
        classify()
