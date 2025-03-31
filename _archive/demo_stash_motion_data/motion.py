# -*- coding: utf-8 -*-
# Time       : 2022/7/20 5:46
# Author     : QIN2DIM
# Github     : https://github.com/QIN2DIM
# Description:
import csv
import os.path
import os.path
import time
from datetime import datetime
from pathlib import Path

from loguru import logger
from sanic import Sanic, Request
from sanic.response import html
from selenium.common.exceptions import WebDriverException
from selenium.webdriver import Chrome, ChromeOptions
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager

app = Sanic("motion-data")


@app.route("/")
async def test(request: Request):
    print(request.headers)
    fp = os.path.join(os.path.dirname(__file__), "motion.html")
    with open(fp, "rb") as file:
        return html(file.read())


class MotionData:
    def __init__(self, temp_dir: Path):
        self.ctx_session = None
        self.temp_dir = temp_dir

        self.action_name = "MotionData"
        self.startpoint = time.time()
        self.sequential_queue = {}

    def __enter__(self):
        options = ChromeOptions()
        service = Service(ChromeDriverManager().install())
        self.ctx_session = Chrome(service=service, options=options)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        logger.debug(
            f">> QUIT [{self.action_name}] Turn off tracker - "
            f"runtime={round(time.time() - self.startpoint, 2)}s"
        )

        try:
            if self.ctx_session:
                self._overload(self.ctx_session)
                self._offload()
                self.ctx_session.quit()
        except AttributeError:
            pass

    def _overload(self, ctx):
        try:
            mouse_track_obj = ctx.find_element(By.CLASS_NAME, "track-coordinate-list")
            mouse_track = mouse_track_obj.text
        except (WebDriverException, AttributeError):
            logger.warning("Failed to record mouse track")
        except Exception as err:
            logger.debug(err)
        else:
            if not mouse_track:
                return
            for p in mouse_track.split(","):
                x = [int(xi.split(".")[0]) for xi in p.split(":")]
                self.sequential_queue[x[0]] = [x[1], x[2]]

    def _offload(self):
        endpoint = (
            str(datetime.utcnow()).replace("-", "").replace(":", "").replace(" ", "").split(".")[0]
        )

        # Record track
        fn = os.path.join(self.temp_dir, "motion_data", f"{endpoint}.csv")
        os.makedirs(os.path.dirname(fn), exist_ok=True)
        with open(fn, "w", encoding="utf8", newline="") as file:
            writer = csv.writer(file)
            for xp, yp in self.sequential_queue.values():
                writer.writerow([xp, yp])

        # Record offset
        fn_offset = os.path.join(os.path.dirname(fn), f"{endpoint}.offset.csv")
        with open(fn_offset, "w", encoding="utf8", newline="") as file:
            writer = csv.writer(file)
            timeline = list(self.sequential_queue)[200:-200]
            for i in range(len(timeline) - 1):
                x1, y1 = self.sequential_queue[timeline[i]]
                x2, y2 = self.sequential_queue[timeline[i + 1]]
                writer.writerow([x2 - x1, y2 - y1])

        logger.debug(
            f">> OFFLOAD [{self.action_name}] Record mouse track - endpoint={endpoint} path={fn}"
        )

    def mimic(self, url: str = "http://127.0.0.1:8000"):
        if not self.ctx_session:
            return

        try:
            self.ctx_session.get(url)
            logger.debug("Press CTRL + C to terminate the action")
            for _ in range(120):
                time.sleep(0.24)
                self._overload(self.ctx_session)
        except (KeyboardInterrupt, EOFError):
            logger.debug("Received keyboard interrupt signal")


def train_motion(test_site: str, temp_dir: Path):
    with MotionData(temp_dir) as motion:
        motion.mimic(test_site)


if __name__ == "__main__":
    train_motion("http://127.0.0.1:8000", temp_dir=Path("temp_dir"))
