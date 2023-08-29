# -*- coding: utf-8 -*-
# Time       : 2023/8/25 14:05
# Author     : QIN2DIM
# GitHub     : https://github.com/QIN2DIM
# Description:
from __future__ import annotations

import hashlib
import json
import random
import shutil
import threading
import time
import uuid
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Literal

from playwright.sync_api import Page, FrameLocator, Response
from playwright.sync_api import TimeoutError as NinjaTimeout

from hcaptcha_challenger.agents.skeleton import Status
from hcaptcha_challenger.components.image_downloader import download_images
from hcaptcha_challenger.components.prompt_handler import split_prompt_message, label_cleaning
from hcaptcha_challenger.onnx.modelhub import ModelHub
from hcaptcha_challenger.onnx.resnet import ResNetControl
from hcaptcha_challenger.onnx.yolo import YOLO_CLASSES, YOLOv8, apply_ash_of_war
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

    def save_example(self, tmp_dir: Path = None):
        shape_type = self.request_config.get("shape_type", "")

        requester_question = self.requester_question.get("en")
        fn = f"{self.request_type}.{shape_type}.{requester_question}.json"
        if tmp_dir and tmp_dir.exists():
            fn = tmp_dir.joinpath(fn)

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
class Radagon:
    page: Page
    """
    Playwright Page
    """

    modelhub: ModelHub
    """
    Build Skeleton with modelhub
    """

    qr: QuestionResp | None = None

    cr: ChallengeResp | None = None

    this_dir: Path = Path(__file__).parent
    """
    Project directory of Skeleton Agents
    """

    tmp_dir: Path = this_dir.joinpath("temp_cache")
    challenge_dir = tmp_dir.joinpath("_challenge")
    """
    Runtime cache
    """

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

    _img_paths: List[Path] = field(default_factory=list)
    """
    bytes of challenge image
    """

    _label = ""
    """
    Cleaned Challenge Prompt in the context
    """

    _prompt = ""
    """
    Challenge Prompt in the context
    """

    label_alias: Dict[str, str] = field(default_factory=dict)
    """
    A collection of { prompt[s]: model_name[.onnx] }
    """

    HOOK_PURCHASE = "//div[@id='webPurchaseContainer']//iframe"
    HOOK_CHECKBOX = "//iframe[contains(@title, 'checkbox for hCaptcha')]"
    HOOK_CHALLENGE = "//iframe[contains(@title, 'hCaptcha challenge')]"

    def __post_init__(self):
        self.handle_question_resp(self.page)
        self.label_alias = self.modelhub.label_alias

    def handler(self, response: Response):
        if response.url.startswith("https://hcaptcha.com/getcaptcha/"):
            with suppress(Exception):
                self.qr = QuestionResp.from_response(response)
                self.qr.save_example(tmp_dir=self.tmp_dir)
        if response.url.startswith("https://hcaptcha.com/checkcaptcha/"):
            with suppress(Exception):
                self.cr = ChallengeResp.from_response(response)

    def handle_question_resp(self, page: Page):
        page.on("response", self.handler)

    @classmethod
    def from_page(cls, page: Page, tmp_dir=None, **kwargs):
        modelhub = ModelHub.from_github_repo(**kwargs)
        modelhub.parse_objects()

        self = cls(page=page, modelhub=modelhub)

        if tmp_dir and isinstance(tmp_dir, Path):
            self.tmp_dir = tmp_dir
            self.challenge_dir = tmp_dir.joinpath("_challenge")

        return self

    @property
    def status(self):
        return Status

    def _switch_to_challenge_frame(self, page: Page, window: str = "login", **kwargs):
        if window == "login":
            frame_challenge = page.frame_locator(self.HOOK_CHALLENGE)
        else:
            frame_purchase = page.frame_locator(self.HOOK_PURCHASE)
            frame_challenge = frame_purchase.frame_locator(self.HOOK_CHALLENGE)

        return frame_challenge

    def _reset_state(self):
        self.cr, self.qr = None, None
        if not self.qr:
            with self.page.expect_response("**getcaptcha**") as resp:
                self.qr = QuestionResp.from_response(resp.value)

    def _get_label(self):
        self._prompt = self.qr.requester_question.get("en")
        _label = split_prompt_message(self._prompt, lang="en")
        self._label = label_cleaning(_label)

    def _download_images(self):
        request_type = self.qr.request_type
        self.typed_dir = self.tmp_dir.joinpath(request_type, self._label)
        self.typed_dir.mkdir(mode=777, parents=True, exist_ok=True)

        # Prelude container
        container = []
        for i, tk in enumerate(self.qr.tasklist):
            challenge_img_path = self.typed_dir.joinpath(f"{time.time()}.{i}.png")
            container.append((challenge_img_path, tk["datapoint_uri"]))

        # Download
        t = threading.Thread(target=download_images, kwargs={"container": container})
        t.start()
        t.join()

        # Optional deduplication
        self._img_paths = []
        for src, _ in container:
            cache = src.read_bytes()
            dst = self.typed_dir.joinpath(f"{hashlib.md5(cache).hexdigest()}.png")
            shutil.move(src, dst)
            self._img_paths.append(dst)

    def _match_solution(self, select: Literal["yolo", "resnet"] = None) -> ResNetControl | YOLOv8:
        """match solution after `tactical_retreat`"""
        focus_label = self.label_alias.get(self._label, "")

        # Match YOLOv8 model
        if not focus_label or select == "yolo":
            answer_keys = list(self.qr.requester_restricted_answer_set.keys())
            ak = answer_keys[0] if len(answer_keys) > 0 else ""
            ash = f"{self._label} {ak}"
            focus_name, yolo_classes = apply_ash_of_war(ash=ash)
            session = self.modelhub.match_net(focus_name=focus_name)
            detector = YOLOv8.from_pluggable_model(session, focus_name)
            return detector

        # Match ResNet model
        focus_name = focus_label
        if not focus_name.endswith(".onnx"):
            focus_name = f"{focus_name}.onnx"
        net = self.modelhub.match_net(focus_name=focus_name)
        control = ResNetControl.from_pluggable_model(net)
        return control

    def _keypoint_challenge(self, frame_challenge: FrameLocator):
        # Load YOLOv8 model from local or remote repo
        detector: YOLOv8 = self._match_solution(select="yolo")

        # Execute the detection task for twice
        times = int(len(self.qr.tasklist))
        for pth in range(times):
            locator = frame_challenge.locator("//div[@class='challenge-view']//canvas")
            locator.wait_for(state="visible")

            path = self.tmp_dir.joinpath("_challenge", f"{uuid.uuid4()}.png")
            locator.screenshot(path=path, type="png")

            # {{< Please click on the X >}}
            res = detector(Path(path), shape_type="point")
            alts = []
            for name, (center_x, center_y), score in res:
                # Bypass unfocused objects
                if name not in self._label:
                    continue
                # Bypass invalid area
                if center_y < 20 or center_y > 520 or center_x < 91 or center_x > 400:
                    continue
                alts.append(
                    {"name": name, "position": {"x": center_x, "y": center_y}, "score": score}
                )

            # Get best result
            alts = sorted(alts, key=lambda x: x["score"])
            best = alts[-1]

            # Click canvas
            locator.click(delay=500, position=best["position"])

            # {{< Verify >}}
            with suppress(NinjaTimeout):
                fl = frame_challenge.locator("//div[@class='button-submit button']")
                fl.click(delay=200)

            # {{< Done | Continue >}}
            if pth == 0:
                self.page.wait_for_timeout(1000)

    def _binary_challenge(self, frame_challenge: FrameLocator):
        classifier = self._match_solution(select="resnet")

        # {{< IMAGE CLASSIFICATION >}}
        times = int(len(self.qr.tasklist) / 9)
        for pth in range(times):
            # Drop element location
            samples = frame_challenge.locator("//div[@class='task-image']")
            count = samples.count()
            # Remember you are human not a robot
            self.page.wait_for_timeout(1700)
            # Classify and Click on the right image
            for i in range(count):
                sample = samples.nth(i)
                sample.wait_for()
                result = classifier.execute(img_stream=self._img_paths[i + pth * 9].read_bytes())
                if result:
                    with suppress(NinjaTimeout):
                        time.sleep(random.uniform(0.1, 0.3))
                        sample.click(delay=200)

            # {{< Verify >}}
            with suppress(NinjaTimeout):
                fl = frame_challenge.locator("//div[@class='button-submit button']")
                fl.click()

            # {{< Done | Continue >}}
            if pth == 0:
                self.page.wait_for_timeout(1000)

    def _is_success(self, page: Page):
        if not self.cr:
            with page.expect_response("**checkcaptcha**") as resp:
                self.cr = ChallengeResp.from_response(resp.value)
        if not self.cr or not self.cr.is_pass:
            return self.status.CHALLENGE_RETRY
        if self.cr.is_pass:
            return self.status.CHALLENGE_SUCCESS


@dataclass
class AgentT(Radagon):
    def __call__(self, *args, **kwargs):
        return self.execute(**kwargs)

    def export_rq(self, record_dir: Path | None = None, flag: str = "") -> Path | None:
        """
          "c":
            "type": "hsw",
            "req": "eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiJ9 ..."
          "is_pass": true,
          "generated_pass_UUID": "P1_eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9 ...",
          "error": ""
        :param record_dir:
        :param flag: filename
        :return:
        """
        # Default output path
        _record_dir = self.tmp_dir
        _flag = f"rqdata-{time.time()}.json"

        # Parse `record_dir` and `flag`
        if isinstance(record_dir, Path) and record_dir.exists():
            _record_dir = record_dir
        if flag and isinstance(flag, str):
            if not flag.endswith(".json"):
                flag = f"{flag}.json"
            _flag = flag

        # Join the path
        _record_dir.joinpath(_flag)
        _rqdata_path = _record_dir.joinpath(_flag)

        _rqdata_path.write_text(json.dumps(self.cr.__dict__, indent=2))

        return _rqdata_path

    def handle_checkbox(self):
        with suppress(TimeoutError):
            checkbox = self.page.frame_locator("//iframe[contains(@title,'checkbox')]")
            checkbox.locator("#checkbox").click()

    def execute(self, **kwargs):
        window = kwargs.get("window", "login")

        # Switch to hCaptcha challenge iframe
        frame_challenge = self._switch_to_challenge_frame(self.page, window)
        self.page.wait_for_timeout(500)

        # Reset state of the OnClick Challenger
        self._reset_state()

        # Parse challenge prompt
        self._get_label()

        # [Optional] Download challenge image as dataset
        # The input of the task should be a screenshot of challenge-view
        # rather than the challenge image itself
        self._download_images()

        # Match Pattern
        if self.qr.request_type == "image_label_binary":
            if not self.label_alias.get(self._label):
                return self.status.CHALLENGE_BACKCALL

            self._binary_challenge(frame_challenge)

        elif self.qr.request_type == "image_label_area_select":
            shape_type = self.qr.request_config.get("shape_type", "")

            # Bypass Low-mAP Detection tasks
            if not any(c in self._label for c in YOLO_CLASSES):
                return self.status.CHALLENGE_BACKCALL

            # Run detection tasks
            if shape_type == "point":
                self._keypoint_challenge(frame_challenge)
            elif shape_type == "bounding_box":
                return self.status.CHALLENGE_BACKCALL

        # Check challenge result
        result = self._is_success(self.page)
        return result
