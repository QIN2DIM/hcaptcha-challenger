# -*- coding: utf-8 -*-
# Time       : 2022/1/16 0:27
# Author     : QIN2DIM
# Github     : https://github.com/QIN2DIM
# Description:
import os
import sys
from typing import Optional

import undetected_chromedriver as uc
from loguru import logger
from selenium.webdriver import ChromeOptions


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


def get_challenge_ctx(silence: Optional[bool] = None, lang: Optional[str] = None):
    """
    Challenger drive for handling human-machine challenges

    :param silence: Control headless browser

    :param lang: Restrict the language of hCatpcha label.
    See https://github.com/QIN2DIM/hcaptcha-challenger/issues/13

    :return:
    """
    # Control headless browser
    silence = True if silence is None or "linux" in sys.platform else silence

    # - Restrict browser startup parameters
    options = ChromeOptions()
    options.add_argument("--log-level=3")
    options.add_argument("--disable-dev-shm-usage")

    # - Restrict the language of hCaptcha label
    # - Environment variables are valid only in the current process
    # and do not affect other processes in the operating system
    os.environ["LANGUAGE"] = "zh" if lang is None else lang
    options.add_argument(f"--lang={os.getenv('LANGUAGE')}")

    logger.debug("🎮 Activate challenger context")
    return uc.Chrome(options=options, headless=silence)
