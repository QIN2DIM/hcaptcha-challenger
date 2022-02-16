# -*- coding: utf-8 -*-
# Time       : 2022/1/16 0:27
# Author     : QIN2DIM
# Github     : https://github.com/QIN2DIM
# Description:
import sys
from typing import Optional

import undetected_chromedriver as uc
from loguru import logger
from selenium.webdriver import ChromeOptions


class ToolBox:
    """可移植的工具箱"""

    @staticmethod
    def init_log(**sink_path):
        """初始化 loguru 日志信息"""
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


def _set_ctx() -> ChromeOptions:
    """统一的 ChromeOptions 启动参数"""
    options = ChromeOptions()
    options.add_argument("--log-level=3")
    options.add_argument("--lang=zh-CN")  # 可能仅在 Windows 生效
    options.add_argument("--disable-dev-shm-usage")
    return options


def get_challenge_ctx(silence: Optional[bool] = None):
    """挑战者驱动 用于处理人机挑战"""

    silence = True if silence is None or "linux" in sys.platform else silence

    # 控制挑战者驱动版本，避免过于超前
    logger.debug("🎮 Activate challenger context")
    return uc.Chrome(options=_set_ctx(), headless=silence)
