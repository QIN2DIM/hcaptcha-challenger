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
import uuid
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, List, Tuple

from loguru import logger
from playwright.sync_api import Page, Response, FrameLocator
from playwright.sync_api import TimeoutError as NinjaTimeout

from hcaptcha_challenger.agents.playwright.binary_challenge import PlaywrightAgent
from hcaptcha_challenger.components.image_downloader import download_images
from hcaptcha_challenger.onnx.yolo import YOLOv8, YOLO_CLASSES
from hcaptcha_challenger.utils import from_dict_to_model


@dataclass
class OnClickResp:
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


@dataclass
class ChallengeResp:
    c: Dict[str, str] = None
    """
    type: hsw
    req: eyj0 ...
    """

    is_pass: bool = None
    """
    true or false
    """

    generated_pass_uuid: str = None
    """
    P1_eyj0 ...
    """

    error: str = None
    """
    Only available if `pass is False`
    """

    @classmethod
    def from_response(cls, response: Response):
        metadata = response.json()
        return cls(
            c=metadata["c"],
            is_pass=metadata["pass"],
            generated_pass_uuid=metadata.get("generated_pass_UUID", ""),
            error=metadata.get("error", ""),
        )


@dataclass
class OnClickAgent(PlaywrightAgent):
    onclick_resp: OnClickResp | None = None
    challenge_resp: ChallengeResp | None = None

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
            if response.url.startswith("https://hcaptcha.com/getcaptcha/"):
                with suppress(Exception):
                    self.onclick_resp = OnClickResp.from_response(response)

        page.on("response", handle)

    def mark_samples(self, frame_challenge: FrameLocator, *args, **kwargs):
        try:
            locator = frame_challenge.locator("//div[@class='challenge-view']//canvas")
            locator.hover()
            request_type = self.onclick_resp.request_type
            self.typed_dir = self.tmp_dir.joinpath(request_type, self._label)
            self.typed_dir.mkdir(mode=777, parents=True, exist_ok=True)
        except Exception as err:
            logger.error(err)

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

    def match_solution(self) -> YOLOv8:
        session = self.modelhub.match_net(focus_name=YOLOv8.best)
        detector = YOLOv8.from_pluggable_model(session)
        return detector

    def challenge(self, frame_challenge: FrameLocator, detector, *args, **kwargs):
        locator = frame_challenge.locator("//div[@class='challenge-view']//canvas")
        locator.wait_for(state="visible")

        page: Page = kwargs.get("page")
        page.wait_for_timeout(1000)

        path = self.tmp_dir.joinpath("_challenge", f"{uuid.uuid4()}.png")
        locator.screenshot(path=path, type="png")

        # {{< Please click on the X >}}
        res = detector(Path(path))
        for name, (center_x, center_y), _ in res:
            # Bypass unfocused objects
            if name not in self._label:
                continue
            # Bypass invalid area
            if center_y < 20 or center_y > 520 or center_x < 91 or center_x > 400:
                continue
            # Click canvas
            position = {"x": center_x, "y": center_y}
            locator.click(delay=500, position=position)
            # print(f">>[{kwargs.get('pth')}] onclick - {name} - {position=}")
            page.wait_for_timeout(500)
            break

        # {{< Verify >}}
        with suppress(NinjaTimeout):
            fl = frame_challenge.locator("//div[@class='button-submit button']")
            fl.click()

    def is_success(
        self, page: Page, frame_challenge: FrameLocator = None, init=True, *args, **kwargs
    ) -> Tuple[str, str]:
        with page.expect_response("**checkcaptcha**") as resp:
            self.challenge_resp = ChallengeResp.from_response(resp.value)
        if not self.challenge_resp:
            return self.status.CHALLENGE_RETRY, "retry"
        if self.challenge_resp.is_pass:
            return self.status.CHALLENGE_SUCCESS, "success"

    def tactical_retreat(self, **kwargs) -> str | None:
        if any(c in self._label for c in YOLO_CLASSES):
            return
        return self.status.CHALLENGE_BACKCALL

    def anti_hcaptcha(
        self, page: Page, window: str = "login", recur_url=None, *args, **kwargs
    ) -> bool | str:
        # Switch to hCaptcha challenge iframe
        frame_challenge = self.switch_to_challenge_frame(page, window)

        # Reset state of the OnClick Challenger
        self.challenge_resp = None
        self.onclick_resp = None
        with page.expect_response("**getcaptcha**") as resp:
            self.onclick_resp = OnClickResp.from_response(resp.value)

        # Parse challenge prompt
        self.get_label(frame_challenge)

        # Bypass `image_label_binary` challenge
        if self.onclick_resp.request_type != "image_label_area_select":
            return self.status.CHALLENGE_BACKCALL

        # [Optional] Download challenge image as dataset
        # The input of the task should be a screenshot of challenge-view
        # rather than the challenge image itself
        self.mark_samples(frame_challenge)
        self.download_images()

        # Bypass Low-mAP Detection tasks
        result = self.tactical_retreat()
        if result in [self.status.CHALLENGE_BACKCALL]:
            return result

        # Load YOLOv8 model from local or remote repo
        detector = self.match_solution()

        # Execute the detection task for twice
        for pth in range(2):
            frame_challenge = self.switch_to_challenge_frame(page, window)
            self.challenge(frame_challenge, detector, pth=pth, page=page)

        # Check challenge result
        with suppress(TypeError):
            result, message = self.is_success(
                page, frame_challenge, window=window, hook_url=recur_url
            )
            if result in [
                self.status.CHALLENGE_SUCCESS,
                self.status.CHALLENGE_CRASH,
                self.status.CHALLENGE_RETRY,
            ]:
                return result
