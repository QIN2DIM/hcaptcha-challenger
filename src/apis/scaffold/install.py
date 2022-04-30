# -*- coding: utf-8 -*-
# Time       : 2022/1/20 16:16
# Author     : QIN2DIM
# Github     : https://github.com/QIN2DIM
# Description:
import sys
import webbrowser
from typing import Optional

from webdriver_manager.chrome import ChromeType
from webdriver_manager.utils import get_browser_version_from_os

from services.hcaptcha_challenger import YOLO, SKRecognition, ElephantsDrawnWithLeaves
from services.settings import DIR_MODEL, logger, PATH_RAINBOW


def _download_model(onnx_prefix: Optional[str] = None):
    """Pull models"""
    logger.debug("Downloading YOLOv5(ONNX) object detection model...")

    YOLO(dir_model=DIR_MODEL, onnx_prefix=onnx_prefix).download_model()
    ElephantsDrawnWithLeaves(dir_model=DIR_MODEL).download_model()


def _download_rainbow():
    """Update rainbow table"""
    logger.debug("Downloading Rainbow Table...")

    SKRecognition().sync_rainbow(path_rainbow=PATH_RAINBOW, convert=True)


def _download_driver():
    """下载浏览器驱动"""
    logger.debug("Detecting google-chrome...")

    # 检测环境变量 `google-chrome`
    browser_version = get_browser_version_from_os(ChromeType.GOOGLE)
    if browser_version != "UNKNOWN":
        return

    # 环境变量中缺少 `google-chrome` 提示玩家手动安装
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


def run(model: Optional[str] = None):
    """下载项目运行所需的各项依赖"""
    _download_model(onnx_prefix=model)
    _download_driver()
    _download_rainbow()
