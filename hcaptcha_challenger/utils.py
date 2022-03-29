# -*- coding: utf-8 -*-
# Time       : 2022/1/16 0:27
# Author     : QIN2DIM
# Github     : https://github.com/QIN2DIM
# Description:
import asyncio
import os
import sys
from typing import Optional, List, Union

import aiohttp
from loguru import logger


class AshFramework:
    """轻量化的协程控件"""

    def __init__(self, docker: Optional[List] = None):
        # 任务容器：queue
        self.worker, self.done = asyncio.Queue(), asyncio.Queue()
        # 任务容器
        self.docker = docker
        # 任务队列满载时刻长度
        self.max_queue_size = 0

    def progress(self) -> str:
        """任务进度"""
        _progress = self.max_queue_size - self.worker.qsize()
        return f"{_progress}/{self.max_queue_size}"

    def preload(self):
        """预处理"""

    def overload(self):
        """任务重载"""
        if self.docker:
            for task in self.docker:
                self.worker.put_nowait(task)
        self.max_queue_size = self.worker.qsize()

    def offload(self) -> Optional[List]:
        """缓存卸载"""
        crash = []
        while not self.done.empty():
            crash.append(self.done.get())
        return crash

    async def control_driver(self, context, session=None):
        """需要并发执行的代码片段"""
        raise NotImplementedError

    async def launcher(self, session=None):
        """适配接口模式"""
        while not self.worker.empty():
            context = self.worker.get_nowait()
            await self.control_driver(context, session=session)

    async def subvert(self, workers: Union[str, int]):
        """
        框架接口

        loop = asyncio.get_event_loop()
        loop.run_until_complete(fl.go(workers))

        :param workers: ["fast", power]
        :return:
        """
        # 任务重载
        self.overload()

        # 弹出空载任务
        if self.max_queue_size == 0:
            return

        # 粘性功率
        workers = self.max_queue_size if workers in ["fast"] else workers
        workers = workers if workers <= self.max_queue_size else self.max_queue_size

        # 弹性分发
        task_list = []
        async with aiohttp.ClientSession() as session:
            for _ in range(workers):
                task = self.launcher(session=session)
                task_list.append(task)
            await asyncio.wait(task_list)


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
    import undetected_chromedriver as uc
    from selenium.webdriver import ChromeOptions
    from webdriver_manager.chrome import ChromeDriverManager

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

    # - Use chromedriver cache to improve application startup speed
    # - Requirement: undetected-chromedriver >= 3.1.5.post2
    driver_executable_path = ChromeDriverManager(log_level=0).install()

    logger.debug("🎮 Activate challenger context")
    return uc.Chrome(
        options=options, headless=silence, driver_executable_path=driver_executable_path
    )
