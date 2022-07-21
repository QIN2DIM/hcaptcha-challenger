# -*- coding: utf-8 -*-
# Time       : 2022/1/16 0:27
# Author     : QIN2DIM
# Github     : https://github.com/QIN2DIM
# Description:
import logging
import os
import sys
from typing import Optional

import undetected_chromedriver as uc
from loguru import logger
from selenium.common.exceptions import WebDriverException
from webdriver_manager.chrome import ChromeDriverManager, ChromeType
from webdriver_manager.core.utils import get_browser_version_from_os


class ToolBox:
    """Portable Toolbox"""

    @staticmethod
    def init_log(**sink_path):
        """Initialize loguru log information"""
        event_logger_format = (
            "<g>{time:YYYY-MM-DD HH:mm:ss}</g> | "
            "<lvl>{level}</lvl> - "
            # "<c><u>{name}</u></c> | "
            "{message}"
        )
        logger.remove()
        logger.add(
            sink=sys.stdout,
            colorize=True,
            level="DEBUG",
            format=event_logger_format,
            diagnose=False,
        )
        if sink_path.get("error"):
            logger.add(
                sink=sink_path.get("error"),
                level="ERROR",
                rotation="1 week",
                encoding="utf8",
                diagnose=False,
            )
        if sink_path.get("runtime"):
            logger.add(
                sink=sink_path.get("runtime"),
                level="DEBUG",
                rotation="20 MB",
                retention="20 days",
                encoding="utf8",
                diagnose=False,
            )
        return logger

    @staticmethod
    def runtime_report(action_name: str, motive: str = "RUN", message: str = "", **params) -> str:
        """格式化输出"""
        flag_ = f">> {motive} [{action_name}]"
        if message != "":
            flag_ += f" {message}"
        if params:
            flag_ += " - "
            flag_ += " ".join([f"{i[0]}={i[1]}" for i in params.items()])

        return flag_


def get_challenge_ctx(silence: Optional[bool] = None, lang: Optional[str] = None):
    """
    Challenger drive for handling human-machine challenges.

    :param silence: Control headless browser
    :param lang: Restrict the language of hCAPTCHA label.
      See https://github.com/QIN2DIM/hcaptcha-challenger/issues/13
    :rtype: uc.Chrome
    """
    # Control headless browser
    silence = True if silence is None or "linux" in sys.platform else silence

    # - Restrict browser startup parameters
    options = uc.ChromeOptions()
    options.add_argument("--log-level=3")
    options.add_argument("--disable-dev-shm-usage")

    # - Restrict the language of hCaptcha label
    # - Environment variables are valid only in the current process
    # and do not affect other processes in the operating system
    os.environ["LANGUAGE"] = "zh_CN" if lang is None else lang
    options.add_argument(f"--lang={os.getenv('LANGUAGE')}")

    if silence is True:
        options.add_argument("--disable-gpu")
        options.add_argument("--disable-software-rasterizer")

    # - Use chromedriver cache to improve application startup speed
    # - Requirement: undetected-chromedriver >= 3.1.5.post2
    logging.getLogger("WDM").setLevel(logging.NOTSET)
    driver_executable_path = ChromeDriverManager().install()
    version_main = get_browser_version_from_os(ChromeType.GOOGLE).split(".")[0]

    logger.debug("🎮 Activate challenger context")
    try:
        return uc.Chrome(
            options=options, headless=silence, driver_executable_path=driver_executable_path
        )
    except WebDriverException:
        return uc.Chrome(
            options=options,
            headless=silence,
            version_main=int(version_main) if version_main.isdigit() else None,
        )
