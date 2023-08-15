# -*- coding: utf-8 -*-
# Time       : 2022/1/16 0:25
# Author     : QIN2DIM
# Github     : https://github.com/QIN2DIM
# Description:

import logging
import os
import sys
import typing
import warnings

from loguru import logger
from selenium.common.exceptions import WebDriverException
from undetected_chromedriver import Chrome, ChromeOptions
from webdriver_manager.chrome import ChromeDriverManager, ChromeType
from webdriver_manager.core.utils import get_browser_version_from_os

warnings.filterwarnings("ignore", category=DeprecationWarning)


def init_log(**sink_channel):
    event_logger_format = "<g>{time:YYYY-MM-DD HH:mm:ss}</g> | <lvl>{level}</lvl> - {message}"
    persistent_format = (
        "<g>{time:YYYY-MM-DD HH:mm:ss}</g> | "
        "<lvl>{level}</lvl>    | "
        "<c><u>{name}</u></c>:{function}:{line} | "
        "{message} - "
        "{extra}"
    )
    serialize_format = event_logger_format + " - {extra}"
    logger.remove()
    logger.add(
        sink=sys.stdout, colorize=True, level="DEBUG", format=serialize_format, diagnose=False
    )
    if sink_channel.get("error"):
        logger.add(
            sink=sink_channel.get("error"),
            level="ERROR",
            rotation="1 week",
            encoding="utf8",
            diagnose=False,
            format=persistent_format,
        )
    if sink_channel.get("runtime"):
        logger.add(
            sink=sink_channel.get("runtime"),
            level="DEBUG",
            rotation="20 MB",
            retention="20 days",
            encoding="utf8",
            diagnose=False,
            format=persistent_format,
        )
    if sink_channel.get("serialize"):
        logger.add(
            sink=sink_channel.get("serialize"),
            level="DEBUG",
            format=persistent_format,
            encoding="utf8",
            diagnose=False,
            serialize=True,
        )
    return logger


class Scaffold:
    """System scaffolding Top-level interface commands"""

    CHALLENGE_LANGUAGE = "en"

    def __init__(self, lang: typing.Optional[str] = None):
        if lang is not None:
            Scaffold.CHALLENGE_LANGUAGE = lang

        self.challenge = self.demo
        self.run = self.demo

    @staticmethod
    def install(model: typing.Optional[str] = "yolov6n", upgrade: typing.Optional[bool] = False):
        """Download Project Dependencies and upgrade all pluggable ONNX model"""

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


def get_challenge_ctx(
    silence: typing.Optional[bool] = None, lang: typing.Optional[str] = None, **kwargs
):
    """
    Challenger drive for handling human-machine challenges.

    :param silence: Control headless browser
    :param lang: Restrict the language of hCAPTCHA label.
      See https://github.com/QIN2DIM/hcaptcha-challenger/issues/13
    :rtype: undetected_chromedriver.Chrome
    """
    # Control headless browser
    # If on Linux, and no X server is available (`DISPLAY` not set), assume
    # headless operation
    silence = (
        True
        if silence is None or ("linux" in sys.platform and "DISPLAY" not in os.environ)
        else silence
    )

    # - Use chromedriver cache to improve application startup speed
    # - Requirement: undetected-chromedriver >= 3.1.5.post2
    logging.getLogger("WDM").setLevel(logging.NOTSET)
    driver_executable_path = ChromeDriverManager().install()
    version_main = get_browser_version_from_os(ChromeType.GOOGLE).split(".")[0]

    logger.debug("🎮 Activate challenger context")
    try:
        return Chrome(
            options=create_chrome_options(silence, lang),
            headless=silence,
            driver_executable_path=driver_executable_path,
            **kwargs,
        )
    except WebDriverException:
        return Chrome(
            options=create_chrome_options(silence, lang),
            headless=silence,
            version_main=int(version_main) if version_main.isdigit() else None,
            **kwargs,
        )


def create_chrome_options(
    silence: typing.Optional[bool] = None, lang: typing.Optional[str] = None
) -> ChromeOptions:
    """
    Create ChromeOptions for undetected_chromedriver.Chrome

    :param silence: Control headless browser
    :param lang: Restrict the language of hCAPTCHA label.

    :rtype: undetected_chromedriver.ChromeOptions
    """

    # - Restrict browser startup parameters
    options = ChromeOptions()
    options.add_argument("--log-level=3")
    options.add_argument("--disable-dev-shm-usage")

    # - Restrict the language of hCaptcha label
    # - Environment variables are valid only in the current process
    # and do not affect other processes in the operating system
    os.environ["LANGUAGE"] = "en_US" if lang is None else lang
    options.add_argument(f"--lang={os.getenv('LANGUAGE')}")

    if silence is True:
        options.add_argument("--disable-gpu")
        options.add_argument("--disable-software-rasterizer")

    return options
