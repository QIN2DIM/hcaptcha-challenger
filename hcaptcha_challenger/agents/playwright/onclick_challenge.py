# -*- coding: utf-8 -*-
# Time       : 2023/8/25 14:05
# Author     : QIN2DIM
# GitHub     : https://github.com/QIN2DIM
# Description:
from __future__ import annotations

import hashlib
import shutil
import threading
import time
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, List

from playwright.sync_api import Page, Response, FrameLocator, Position

from hcaptcha_challenger.agents.playwright.binary_challenge import PlaywrightAgent
from hcaptcha_challenger.components.image_downloader import download_images
from hcaptcha_challenger.onnx.yolo import YOLOv8
from hcaptcha_challenger.utils import from_dict_to_model


@dataclass
class ChallengeResp:
    c: Dict[str, str] = field(default_factory=dict)
    """
    type: hsw
    req: eyJ0...
    """

    challenge_uri: str = ""
    """
    "https://hcaptcha.com/challenge/selection/challenge.js
    """

    key: str = ""
    """
    E0_ey xxxx
    """

    request_config: Dict[str, Any] = field(default_factory=dict)

    request_type: str = ""
    """
    1. image_label_binary
    2. image_label_area_select
    """

    requester_question: Dict[str, str] = field(default_factory=dict)
    """
    "en": "Please click on the rac\u0441oon"
    """

    requester_question_example: List[str] = field(default_factory=list)
    """
    List of urls: https://imgs.hcaptcha.com/ + base64
    """

    requester_restricted_answer_set: Dict[dict] = field(default_factory=dict)
    """
    Not available on the binary challenge
    """

    tasklist: List[Dict[str, str]] = field(default_factory=list)
    """
    {
      "datapoint_uri": "https://imgs.hcaptcha.com/6PHAmcW7GU_mnuCwlK0_sxcNKLAx-8SfsyyW_6NnDPf6YUedUe64oiTpDsnWivBxQMhPYDcHcoq38lohjyvWammOgLPcE9sEY4AqnEFEaGgOYx1cZEgbc8hdm69k3tIIykEGuMGopOD9SQST1muhSL-NuiTvsMIPQf2F_p7965ErmqTxR1MbuW2JTwT0ka5Zl-o08ger0gx9okfr4SPluYFL7e2Jl2WkWCmO1OUsnSLthP07re2tSz0v1W40CWp3IxPq5kOAVyyTLEEKgYJaqwP1PBz0EugpQ1nNn8Ut",
      "task_key": "c1cdadfb-3ad0-4f61-9260-c3827c76e92c"
    }
    """

    @classmethod
    def from_response(cls, response: Response):
        return from_dict_to_model(cls, response.json())

    def get_datapoint(self, pth: int):
        return self.tasklist[pth]["datapoint_uri"]


@dataclass
class OnClickAgent(PlaywrightAgent):
    onclick_resp: ChallengeResp = None

    typed_dir: Path = Path("please click on the X")
    """
    - image_label_area_select
        - please click on the X #typed
            - hash_md5.png
            - xxx
    """

    img_paths: List[Path] = field(default_factory=list)
    """
    bytes of challenge image
    """

    def handle_onclick_resp(self, page: Page):
        def handle(response: Response):
            if not response.url.startswith("https://hcaptcha.com/getcaptcha/"):
                return
            with suppress(Exception):
                self.onclick_resp = ChallengeResp.from_response(response)

        page.on("response", handle)

    def mark_samples(self, frame_challenge: FrameLocator, *args, **kwargs):
        try:
            locator = frame_challenge.locator("//div[@class='challenge-view']//canvas")
            locator.hover()
            request_type = self.onclick_resp.request_type
            self.typed_dir = self.tmp_dir.joinpath(request_type, self._label)
            self.typed_dir.mkdir(mode=777, parents=True, exist_ok=True)
        except Exception as err:
            print(err)

    def download_images(self):
        # Prelude container
        container = []
        for i, tk in enumerate(self.onclick_resp.tasklist):
            challenge_img_path = self.typed_dir.joinpath(f"{time.time()}.{i}.png")
            container.append((challenge_img_path, tk["datapoint_uri"]))

        # Download
        t = threading.Thread(target=download_images, kwargs={"container": container})
        t.start()
        t.join()

        # Optional deduplication
        for src, _ in container:
            cache = src.read_bytes()
            dst = self.typed_dir.joinpath(f"{hashlib.md5(cache).hexdigest()}.png")
            shutil.move(src, dst)
            self.img_paths.append(dst)

    def anti_hcaptcha(
            self, page: Page, window: str = "login", recur_url=None, *args, **kwargs
    ) -> bool | str:
        frame_challenge = self.switch_to_challenge_frame(page, window)
        page.wait_for_timeout(2000)

        self.get_label(frame_challenge)

        if self.onclick_resp.request_type != "image_label_area_select":
            return self.status.CHALLENGE_BACKCALL

        self.mark_samples(frame_challenge)

        self.download_images()

        model_path = self.modelhub.models_dir.joinpath("onclick_yolov8n.onnx")
        detector = YOLOv8.from_model_path(model_path)
        res = detector(self.img_paths[0])
        for name, (center_x, center_y), _ in res:
            if name not in self._label:
                continue
            print(name, center_x, center_y)
            locator = frame_challenge.locator("//div[@class='challenge-view']//canvas")
            locator.hover(position=Position(x=center_x, y=center_y))
            locator.click(delay=1000)
        input("123")
        return self.status.CHALLENGE_BACKCALL
