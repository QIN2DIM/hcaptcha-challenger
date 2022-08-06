# -*- coding: utf-8 -*-
# Time       : 2022/1/20 16:16
# Author     : QIN2DIM
# Github     : https://github.com/QIN2DIM
# Description:
import hashlib
import os
import sys
import webbrowser
from typing import Optional

from webdriver_manager.chrome import ChromeType
from webdriver_manager.core.utils import get_browser_version_from_os

from services.hcaptcha_challenger import YOLO, SKRecognition, PluggableONNXModels
from services.settings import DIR_MODEL, logger, PATH_RAINBOW_YAML, PATH_OBJECTS_YAML


def download_driver():
    # Detect environment variable `google-chrome`.
    browser_version = get_browser_version_from_os(ChromeType.GOOGLE)
    if browser_version != "UNKNOWN":
        return

    # `google-chrome` is missing from environment variables, prompting players to install manually.
    logger.critical(
        "The current environment variable is missing `google-chrome`, "
        "please install Chrome for your system"
    )
    logger.info(
        "Ubuntu: https://linuxize.com/post/how-to-install-google-chrome-web-browser-on-ubuntu-20-04/"
    )
    logger.info(
        "CentOS 7/8: https://linuxize.com/post/how-to-install-google-chrome-web-browser-on-centos-7/"
    )
    if "linux" not in sys.platform:
        webbrowser.open("https://www.google.com/chrome/")

    logger.info("Re-execute the `install` scaffolding command after the installation is complete.")


def download_yolo_model(onnx_prefix):
    YOLO(dir_model=DIR_MODEL, onnx_prefix=onnx_prefix).download_model()


def refresh_pluggable_onnx_model(upgrade: Optional[bool] = None):
    def need_to_refresh():
        _flag = "5ba2edb8fdd1350ff3a8731bc71c313998ba70a32d58178a218545e54d2701cf"
        if not os.path.exists(PATH_RAINBOW_YAML):
            return True
        with open(PATH_RAINBOW_YAML, "rb") as file:
            return hashlib.sha256(file.read()).hexdigest() != _flag

    if need_to_refresh():
        SKRecognition().sync_rainbow(path_rainbow=PATH_RAINBOW_YAML, convert=True)
        PluggableONNXModels(PATH_OBJECTS_YAML).summon(dir_model=DIR_MODEL, upgrade=upgrade)


def run(model: Optional[str] = None, upgrade: Optional[bool] = None):
    """下载项目运行所需的各项依赖"""
    download_driver()
    download_yolo_model(onnx_prefix=model)
    refresh_pluggable_onnx_model(upgrade=upgrade)
