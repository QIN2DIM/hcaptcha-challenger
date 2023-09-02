# -*- coding: utf-8 -*-
# Time       : 2023/8/25 14:05
# Author     : QIN2DIM
# GitHub     : https://github.com/QIN2DIM
# Description:
from __future__ import annotations

import asyncio
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

from playwright.async_api import Page, FrameLocator, Response
from playwright.async_api import TimeoutError

from hcaptcha_challenger.agents.skeleton import Status
from hcaptcha_challenger.components.image_downloader import download_images
from hcaptcha_challenger.components.prompt_handler import split_prompt_message, label_cleaning
from hcaptcha_challenger.onnx.modelhub import ModelHub
from hcaptcha_challenger.onnx.resnet import ResNetControl
from hcaptcha_challenger.onnx.yolo import YOLOv8, is_matched_area_select_label, finetune_keypoint
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

    task_list: List[Dict[str, str]] = field(default_factory=list)
    """
    [
        {datapoint_uri: "https://imgs.hcaptcha.com + base64", task_key: "" },
        {datapoint_uri: "https://imgs.hcaptcha.com + base64", task_key: "" },
    ]
    """

    @classmethod
    def from_json(cls, data: Dict[str, Any]):
        return from_dict_to_model(cls, data)

    def save_example(self, tmp_dir: Path = None):
        shape_type = self.request_config.get("shape_type", "")

        requester_question = self.requester_question.get("en")
        file_name = f"{self.request_type}.{shape_type}.{requester_question}.json"
        if tmp_dir and tmp_dir.exists():
            file_name = tmp_dir.joinpath(file_name)

        Path(file_name).write_text(json.dumps(self.__dict__, indent=2))


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
    def from_json(cls, metadata: Dict[str, Any]):
        return cls(
            c=metadata["c"],
            is_pass=metadata["pass"],
            generated_pass_UUID=metadata.get("generated_pass_UUID", ""),
            error=metadata.get("error", ""),
        )


@dataclass
class Challenger:
    page: Page
    """
    Playwright Page
    """

    modelhub: ModelHub
    """
    Build Skeleton with modelhub
    """

    quest_resp: QuestionResp | None = None
    challenge_resp: ChallengeResp | None = None

    quest_resp_queue: asyncio.Queue[QuestionResp] | None = None
    challenge_resp_queue: asyncio.Queue[ChallengeResp] | None = None

    current_dir: Path = Path(__file__).parent
    """
    Project directory of Skeleton Agents
    """

    tmp_dir: Path = current_dir.joinpath("temp_cache")
    challenge_dir = tmp_dir.joinpath("_challenge")
    """
    Runtime cache
    """

    type_dir: Path = Path("please click on the X")
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
        self.label_alias = self.modelhub.label_alias

        self.quest_resp_queue = asyncio.Queue()
        self.challenge_resp_queue = asyncio.Queue()

        self.handle_question_resp(self.page)

    async def handler(self, response: Response):
        if response.url.startswith("https://hcaptcha.com/getcaptcha/"):
            with suppress(Exception):
                data = await response.json()
                quest_resp = QuestionResp.from_json(data)
                quest_resp.save_example(tmp_dir=self.tmp_dir)
                self.quest_resp_queue.put_nowait(quest_resp)
        if response.url.startswith("https://hcaptcha.com/checkcaptcha/"):
            with suppress(Exception):
                metadata = await response.json()
                challenge_resp = ChallengeResp.from_json(metadata)
                self.challenge_resp_queue.put_nowait(challenge_resp)

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

    @property
    def area_select_question(self):
        answer_keys = list(self.quest_resp.requester_restricted_answer_set.keys())
        answer_key = answer_keys[0] if len(answer_keys) > 0 else ""
        area_select_question = f"{self._label} {answer_key}"
        return area_select_question

    def _switch_to_challenge_frame(self, page: Page, window: str = "login", **kwargs):
        if window == "login":
            frame_challenge = page.frame_locator(self.HOOK_CHALLENGE)
        else:
            frame_purchase = page.frame_locator(self.HOOK_PURCHASE)
            frame_challenge = frame_purchase.frame_locator(self.HOOK_CHALLENGE)

        return frame_challenge

    async def _reset_state(self):
        self.challenge_resp = None
        self.quest_resp = await self.quest_resp_queue.get()

    def _parse_label(self):
        self._prompt = self.quest_resp.requester_question.get("en")
        _label = split_prompt_message(self._prompt, lang="en")
        self._label = label_cleaning(_label)

    def _download_images(self):
        request_type = self.quest_resp.request_type
        self.type_dir = self.tmp_dir.joinpath(request_type, self._label)
        self.type_dir.mkdir(parents=True, exist_ok=True)

        container = []
        for i, tk in enumerate(self.quest_resp.task_list):
            challenge_img_path = self.type_dir.joinpath(f"{time.time()}.{i}.png")
            container.append((challenge_img_path, tk["datapoint_uri"]))

        t = threading.Thread(target=download_images, kwargs={"container": container})
        t.start()
        t.join()

        # Optional deduplication
        self._img_paths = []
        for src, _ in container:
            cache = src.read_bytes()
            dst = self.type_dir.joinpath(f"{hashlib.md5(cache).hexdigest()}.png")
            shutil.move(src, dst)
            self._img_paths.append(dst)

    def _match_solution(self, select: Literal["yolo", "resnet"] = None) -> ResNetControl | YOLOv8:
        """match solution after `retreat_challenge`"""
        focus_label = self.label_alias.get(self._label, "")

        # Match YOLOv8 model
        if not focus_label or select == "yolo":
            focus_name, classes = self.modelhub.apply_area_select_label(area_select_question=self.area_select_question)
            session = self.modelhub.match_net(focus_name=focus_name)
            detector = YOLOv8.from_pluggable_model(session, classes)
            return detector

        # Match ResNet model
        focus_name = focus_label
        if not focus_name.endswith(".onnx"):
            focus_name = f"{focus_name}.onnx"
        net = self.modelhub.match_net(focus_name=focus_name)
        control = ResNetControl.from_pluggable_model(net)
        return control

    async def _bounding_challenge(self, frame_challenge: FrameLocator):
        detector: YOLOv8 = self._match_solution(select="yolo")
        times = int(len(self.quest_resp.task_list))
        for pth in range(times):
            locator = frame_challenge.locator("//div[@class='challenge-view']//canvas")
            await locator.wait_for(state="visible")

            path = self.tmp_dir.joinpath("_challenge", f"{uuid.uuid4()}.png")
            await locator.screenshot(path=path, type="png")

            res = detector(Path(path), shape_type="bounding_box")

            alts = []
            for name, (x1, y1), (x2, y2), score in res:
                if not is_matched_area_select_label(area_select_question=self.area_select_question, class_name=name):
                    continue

                alt = {"name": name,
                       "start": (int(x1), int(y1)),
                       "end": (int(x2), int(y2)),
                       "scoop": (x2 - x1) * (y2 - y1)}
                alts.append(alt)

            if len(alts) > 1:
                alts = sorted(alts, key=lambda _alt: _alt["scoop"])
            if len(alts) > 0:
                best = alts[-1]
                x1, y1 = best["start"]
                x2, y2 = best["end"]
                await locator.click(delay=200, position={"x": x1, "y": y1})
                await self.page.mouse.move(x2, y2)
                await locator.click(delay=200, position={"x": x2, "y": y2})

            with suppress(TimeoutError):
                submit_button = frame_challenge.locator("//div[@class='button-submit button']")
                await submit_button.click(delay=200)

            if pth == 0:
                await self.page.wait_for_timeout(1000)

    async def _keypoint_challenge(self, frame_challenge: FrameLocator):
        # Load YOLOv8 model from local or remote repo
        detector: YOLOv8 = self._match_solution(select="yolo")

        # Execute the detection task for twice
        times = int(len(self.quest_resp.task_list))
        for pth in range(times):
            locator = frame_challenge.locator("//div[@class='challenge-view']//canvas")
            await locator.wait_for(state="visible")

            path = self.tmp_dir.joinpath("_challenge", f"{uuid.uuid4()}.png")
            await locator.screenshot(path=path, type="png")

            # {{< Please click on the X >}}
            res = detector(Path(path), shape_type="point")
            # print(res)

            alts = []
            for name, (center_x, center_y), score in res:
                # Bypass unfocused objects
                if not is_matched_area_select_label(area_select_question=self.area_select_question, class_name=name):
                    continue
                # Bypass invalid area
                if center_y < 20 or center_y > 520 or center_x < 91 or center_x > 400:
                    continue
                center_x, center_y = finetune_keypoint(name, [center_x, center_y])
                alt = {"name": name, "position": {"x": center_x, "y": center_y}, "score": score}
                alts.append(alt)

            # Get best result
            if len(alts) > 1:
                alts = sorted(alts, key=lambda x: x["score"])
            # Click canvas
            if len(alts) > 0:
                best = alts[-1]
                await locator.click(delay=500, position=best["position"])
                # print(f">> Click on the object - position={best['position']} name={best['name']}")
            # Catch-all rule
            else:
                await locator.click(delay=500)
                # print(">> click on the center of the canvas")

            # {{< Verify >}}
            with suppress(TimeoutError):
                fl = frame_challenge.locator("//div[@class='button-submit button']")
                await fl.click(delay=200)

            # {{< Done | Continue >}}
            if pth == 0:
                await self.page.wait_for_timeout(1000)

    async def _binary_challenge(self, frame_challenge: FrameLocator):
        classifier = self._match_solution(select="resnet")

        # {{< IMAGE CLASSIFICATION >}}
        times = int(len(self.quest_resp.task_list) / 9)
        for pth in range(times):
            # Drop element location
            samples = frame_challenge.locator("//div[@class='task-image']")
            count = await samples.count()
            # Remember you are human not a robot
            await self.page.wait_for_timeout(1700)
            # Classify and Click on the right image
            for i in range(count):
                sample = samples.nth(i)
                await sample.wait_for()
                result = classifier.execute(img_stream=self._img_paths[i + pth * 9].read_bytes())
                if result:
                    with suppress(TimeoutError):
                        time.sleep(random.uniform(0.1, 0.3))
                        await sample.click(delay=200)

            # {{< Verify >}}
            with suppress(TimeoutError):
                submit_button = frame_challenge.locator("//div[@class='button-submit button']")
                await submit_button.click()

            # {{< Done | Continue >}}
            if pth == 0:
                await self.page.wait_for_timeout(1000)

    async def _is_success(self):
        self.challenge_resp = await self.challenge_resp_queue.get()
        if not self.challenge_resp or not self.challenge_resp.is_pass:
            return self.status.CHALLENGE_RETRY
        if self.challenge_resp.is_pass:
            return self.status.CHALLENGE_SUCCESS


@dataclass
class CaptchaAgent(Challenger):
    rqdata: ChallengeResp = None

    async def __call__(self, *args, **kwargs):
        return await self.execute(**kwargs)

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
        if not self.challenge_resp:
            return

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

        _rqdata_path.write_text(json.dumps(self.challenge_resp.__dict__, indent=2))

        return _rqdata_path

    async def handle_checkbox(self):
        with suppress(TimeoutError):
            checkbox = self.page.frame_locator("//iframe[contains(@title,'checkbox')]")
            await checkbox.locator("#checkbox").click()

    async def execute(self, **kwargs):
        window = kwargs.get("window", "login")

        frame_challenge = self._switch_to_challenge_frame(self.page, window)

        await self._reset_state()

        # Match: ChallengePassed
        if not self.quest_resp.requester_question.keys():
            return self.status.CHALLENGE_SUCCESS

        self._parse_label()

        self._download_images()

        # Match: image_label_binary
        if self.quest_resp.request_type == "image_label_binary":
            if not self.label_alias.get(self._label):
                return self.status.CHALLENGE_BACKCALL
            await self._binary_challenge(frame_challenge)
        # Match: image_label_area_select
        elif self.quest_resp.request_type == "image_label_area_select":
            area_select_question = self.area_select_question
            if not any(is_matched_area_select_label(area_select_question, c) for c in self.modelhub.yolo_names):
                return self.status.CHALLENGE_BACKCALL

            shape_type = self.quest_resp.request_config.get("shape_type", "")
            if shape_type == "point":
                await self._keypoint_challenge(frame_challenge)
            elif shape_type == "bounding_box":
                await self._bounding_challenge(frame_challenge)

        result = await self._is_success()
        return result

    async def collect(self):
        """Download datasets"""
        await self._reset_state()
        if not self.quest_resp.requester_question.keys():
            return self._label
        self._parse_label()
        self._download_images()
        return self._label
