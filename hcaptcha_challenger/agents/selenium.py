# -*- coding: utf-8 -*-
# Time       : 2023/8/15 20:41
# Author     : QIN2DIM
# GitHub     : https://github.com/QIN2DIM
# Description:
from __future__ import annotations

import logging
import os
import sys
import typing
import warnings
from dataclasses import dataclass
from typing import Tuple

from loguru import logger
from selenium.common.exceptions import (
    WebDriverException,
    TimeoutException,
    NoSuchElementException,
    ElementNotInteractableException,
)
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.wait import WebDriverWait
from undetected_chromedriver import Chrome
from undetected_chromedriver import ChromeOptions
from webdriver_manager.chrome import ChromeDriverManager

from hcaptcha_challenger.agents.skeleton import Skeleton

warnings.filterwarnings("ignore", category=DeprecationWarning)


@dataclass
class SeleniumAgent(Skeleton):
    def switch_to_challenge_frame(self, ctx, **kwargs):
        pass

    def get_label(self, ctx, **kwargs):
        pass

    def mark_samples(self, ctx, *args, **kwargs):
        pass

    def challenge(self, ctx, model, *args, **kwargs):
        pass

    def is_success(self, ctx, *args, **kwargs) -> Tuple[str, str]:
        pass

    def anti_checkbox(self, ctx, *args, **kwargs):
        pass

    def anti_hcaptcha(self, ctx, *args, **kwargs) -> bool | str:
        pass


class ArmorUtils:
    @staticmethod
    def face_the_checkbox(ctx: Chrome) -> bool | None:
        try:
            WebDriverWait(ctx, 8, ignored_exceptions=(WebDriverException,)).until(
                EC.presence_of_element_located((By.XPATH, "//iframe[contains(@title,'checkbox')]"))
            )
            return True
        except TimeoutException:
            return False

    @staticmethod
    def get_hcaptcha_response(ctx: Chrome) -> str | None:
        return ctx.execute_script("return hcaptcha.getResponse()")

    @staticmethod
    def refresh(ctx: Chrome) -> bool | None:
        try:
            ctx.find_element(By.XPATH, "//div[@class='refresh button']").click()
        except (NoSuchElementException, ElementNotInteractableException):
            return False
        return True


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

    logger.debug("🎮 Activate challenger context")
    return Chrome(
        options=create_chrome_options(silence, lang),
        headless=silence,
        driver_executable_path=driver_executable_path,
        **kwargs,
    )
