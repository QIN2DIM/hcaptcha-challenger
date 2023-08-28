# -*- coding: utf-8 -*-
# Time       : 2023/8/19 17:17
# Author     : QIN2DIM
# GitHub     : https://github.com/QIN2DIM
# Description:
from __future__ import annotations

import hashlib
import json
import random
import re
import shutil
import threading
import time
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple, Dict, Any, List, Iterable

from playwright.sync_api import FrameLocator, Page
from playwright.sync_api import Response
from playwright.sync_api import TimeoutError as NinjaTimeout

from hcaptcha_challenger.agents.skeleton import Skeleton
from hcaptcha_challenger.components.image_downloader import download_images
from hcaptcha_challenger.components.prompt_handler import split_prompt_message, label_cleaning
from hcaptcha_challenger.utils import from_dict_to_model


@dataclass
class QuestionResp:
    c: Dict[str, str] = field(default_factory=dict)
    """
    type: hsw
    req: eyj0 ...
    """

    challenge_uri: str = ""
    """
    https://hcaptcha.com/challenge/grid/challenge.js
    """

    key: str = ""
    """
    E0_eyj0 ...
    """

    request_config: Dict[str, Any] = field(default_factory=dict)

    request_type: str = ""
    """
    1. image_label_binary
    2. image_label_area_select  
    """

    requester_question: Dict[str, str] = field(default_factory=dict)
    """
    image_label_binary      | { en: Please click on all images containing an animal }
    image_label_area_select | { en: Please click on the rac\u0441oon }
    """

    requester_question_example: List[str] = field(default_factory=list)
    """
    [
        "https://imgs.hcaptcha.com/ + base64"
    ]
    """

    requester_restricted_answer_set: Dict[dict] = field(default_factory=dict)
    """
    Not available on the binary challenge
    """

    tasklist: List[Dict[str, str]] = field(default_factory=list)
    """
    [
        {datapoint_uri: "https://imgs.hcaptcha.com + base64", task_key: "" },
        {datapoint_uri: "https://imgs.hcaptcha.com + base64", task_key: "" },
    ]
    """

    @classmethod
    def from_response(cls, response: Response):
        return from_dict_to_model(cls, response.json())

    def save_example(self):
        shape_type = self.request_config.get("shape_type", "")
        fn = f"{self.request_type}.{shape_type}.json"
        Path(fn).write_text(json.dumps(self.__dict__, indent=2))


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

    generated_pass_UUID: str = None
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
            generated_pass_UUID=metadata.get("generated_pass_UUID", ""),
            error=metadata.get("error", ""),
        )


@dataclass
class PlaywrightAgent(Skeleton):
    question_resp: QuestionResp | None = None

    challenge_resp: ChallengeResp | None = None

    typed_dir: Path = Path("please click on the X")
    """
    - image_label_area_select
        - please click on the X  # typed
            - hash_md5.png
            - xxx
    - image_label_binary
        - dolphin  # typed
            - hash_md5.png
            - xxx
    """

    img_paths: List[Path] = field(default_factory=list)
    """
    bytes of challenge image
    """

    # //iframe[@id='talon_frame_checkout_free_prod']
    HOOK_PURCHASE = "//div[@id='webPurchaseContainer']//iframe"
    HOOK_CHECKBOX = "//iframe[contains(@title, 'checkbox for hCaptcha')]"
    HOOK_CHALLENGE = "//iframe[contains(@title, 'hCaptcha challenge')]"

    critical_threshold = 3

    def handler(self, response: Response):
        if response.url.startswith("https://hcaptcha.com/getcaptcha/"):
            with suppress(Exception):
                self.question_resp = QuestionResp.from_response(response)
                self.question_resp.save_example()
        if response.url.startswith("https://hcaptcha.com/checkcaptcha/"):
            with suppress(Exception):
                self.challenge_resp = ChallengeResp.from_response(response)

    def handle_question_resp(self, page: Page):
        page.on("response", self.handler)

    def switch_to_challenge_frame(self, page: Page, window: str = "login", **kwargs):
        if window == "login":
            frame_challenge = page.frame_locator(self.HOOK_CHALLENGE)
        else:
            frame_purchase = page.frame_locator(self.HOOK_PURCHASE)
            frame_challenge = frame_purchase.frame_locator(self.HOOK_CHALLENGE)

        return frame_challenge

    def get_label(self, frame_challenge: FrameLocator, **kwargs):
        self._prompt = self.question_resp.requester_question.get("en")
        _label = split_prompt_message(self._prompt, lang="en")
        self._label = label_cleaning(_label)

    def mark_samples(self, frame_challenge: FrameLocator, *args, **kwargs):
        """Get the download link and locator of each challenge image"""
        samples = frame_challenge.locator("//div[@class='task-image']")
        count = samples.count()
        for i in range(count):
            sample = samples.nth(i)
            sample.wait_for()
            alias = sample.get_attribute("aria-label")
            image_style = sample.locator(".image").get_attribute("style")
            url = re.split(r'[(")]', image_style)[2]
            self._alias2url.update({alias: url})
            self._alias2locator.update({alias: sample})

    def download_images(self):
        request_type = self.question_resp.request_type
        self.typed_dir = self.tmp_dir.joinpath(request_type, self._label)
        self.typed_dir.mkdir(mode=777, parents=True, exist_ok=True)

        # Prelude container
        container = []
        for i, tk in enumerate(self.question_resp.tasklist):
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

    def challenge(self, frame_challenge: FrameLocator, model, *args, **kwargs):
        # {{< IMAGE CLASSIFICATION >}}
        for alias, path in self._alias2path.items():
            result = model.execute(img_stream=path.read_bytes())
            if result:
                with suppress(NinjaTimeout):
                    time.sleep(random.uniform(0.1, 0.3))
                    self._alias2locator[alias].click()

        # {{< SUBMIT ANSWER >}}
        with suppress(NinjaTimeout):
            fl = frame_challenge.locator("//div[@class='button-submit button']")
            fl.click(delay=1000, timeout=5000)

    def is_success(
        self, page: Page, frame_challenge: FrameLocator = None, init=True, *args, **kwargs
    ) -> Tuple[str, str]:
        if not self.challenge_resp:
            with page.expect_response("**checkcaptcha**") as resp:
                self.challenge_resp = ChallengeResp.from_response(resp.value)
        if not self.challenge_resp:
            return self.status.CHALLENGE_RETRY, "retry"
        if self.challenge_resp.is_pass:
            return self.status.CHALLENGE_SUCCESS, "success"

    def anti_checkbox(self, page: Page, *args, **kwargs):
        checkbox = page.frame_locator("//iframe[contains(@title,'checkbox')]")
        checkbox.locator("#checkbox").click()

    def anti_hcaptcha(
        self, page: Page, window: str = "login", recur_url=None, *args, **kwargs
    ) -> bool | str:
        # Switch to hCaptcha challenge iframe
        frame_challenge = self.switch_to_challenge_frame(page, window)

        # Execute the classification task for twice
        for pth in range(2):
            # Reset state of the OnClick Challenger
            self.challenge_resp = None
            self.question_resp = None
            with page.expect_response("**getcaptcha**") as resp:
                self.question_resp = QuestionResp.from_response(resp.value)

            # Parse challenge prompt
            self.get_label(frame_challenge)

            # Download challenge image as dataset
            self.mark_samples(frame_challenge)
            self.download_images()

            # Bypass Low-mAP classification tasks
            if not self._label_alias.get(self._label):
                return self.status.CHALLENGE_BACKCALL

            # Load ResNet model from local or remote repo
            classifier = self.match_solution(select="resnet")

            # Execute the classification task for twice
            frame_challenge = self.switch_to_challenge_frame(page, window)
            self.challenge(frame_challenge, model=classifier)

            # Check challenge result
            result = self.is_success(page, frame_challenge, window=window, hook_url=recur_url)
            if isinstance(result, Iterable):
                result = result[0]
            return result
