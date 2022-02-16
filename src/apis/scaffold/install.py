# -*- coding: utf-8 -*-
# Time       : 2022/1/20 16:16
# Author     : QIN2DIM
# Github     : https://github.com/QIN2DIM
# Description:
import sys
import webbrowser

from webdriver_manager.utils import get_browser_version_from_os

from services.hcaptcha_challenger import YOLO
from services.settings import DIR_MODEL, logger
from services.utils import get_challenge_ctx


def _download_model(onnx_prefix="yolov5s6"):
    """下载 YOLOv4 目标检测模型"""
    logger.debug("Downloading YOLOv5 object detection model...")

    YOLO(dir_model=DIR_MODEL,onnx_prefix=onnx_prefix).download_model()


def _download_driver():
    """下载浏览器驱动"""
    logger.debug("Detecting google-chrome...")

    # 检测环境变量 `google-chrome`
    browser_version = get_browser_version_from_os("google-chrome")
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

    logger.info(
        "Re-execute the `install` scaffolding command after the installation is complete."
    )


def run(model:str="yolov5s6"):
    """下载项目运行所需的各项依赖"""
    _download_model(onnx_prefix=model)
    _download_driver()


@logger.catch()
def test():
    """检查挑战者驱动版本是否适配"""
    ctx = get_challenge_ctx(silence=True)
    try:
        ctx.get("https://www.baidu.com")
    finally:
        ctx.quit()

    logger.success("The adaptation is successful")
