# -*- coding: utf-8 -*-
# Time       : 2022/7/20 5:41
# Author     : QIN2DIM
# Github     : https://github.com/QIN2DIM
# Description:
import os.path
import time

from selenium.common.exceptions import WebDriverException
from selenium.webdriver import Chrome, ChromeOptions
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager

from services.settings import logger
from services.utils import ToolBox


class MotionData:
    def __init__(self, dir_database: str = None):
        self.ctx_session = None
        self.dir_log = "tracker" if dir_database is None else dir_database

        self.action_name = "MotionData"
        self.offload_count = 0

    def __enter__(self):
        options = ChromeOptions()
        service = Service(ChromeDriverManager().install())
        self.ctx_session = Chrome(service=service, options=options)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            if self.ctx_session:
                self.ctx_session.quit()
        except AttributeError:
            pass

    def _offload(self, ctx):
        try:
            mouse_track: str = ctx.find_element(By.CLASS_NAME, "track-coordinate-list").text
        except (WebDriverException, AttributeError):
            logger.warning("Failed to record mouse track")
        else:
            endpoint = int(time.time())
            fn = os.path.join(self.dir_log, "motion_data", f"{endpoint}.txt")
            os.makedirs(os.path.dirname(fn), exist_ok=True)
            with open(fn, "w", encoding="utf8") as file:
                file.write(mouse_track.replace(",", "\n"))
            self.offload_count += 1
            logger.success(
                ToolBox.runtime_report(
                    motive="OFFLOAD",
                    action_name=self.action_name,
                    message="Record mouse track",
                    endpoint=endpoint,
                    path=fn,
                )
            )

    def mimic(self, url: str = "http://127.0.0.1:8000"):
        if not self.ctx_session:
            return
        self.ctx_session.get(url)
        start = time.time()
        try:
            while True:
                time.sleep(5)
                self._offload(self.ctx_session)
        except (KeyboardInterrupt, EOFError):
            logger.debug(
                ToolBox.runtime_report(
                    motive="QUIT",
                    action_name=self.action_name,
                    message="Turn off tracker",
                    record_count=self.offload_count,
                    runtime=f"{round(time.time() - start, 2)}s",
                )
            )
