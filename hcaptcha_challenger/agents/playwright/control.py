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
import time
import uuid
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Literal

from loguru import logger
from playwright.async_api import Page, FrameLocator, Response, Position
from playwright.async_api import TimeoutError

from hcaptcha_challenger.components.cv_toolkit import find_unique_object, annotate_objects
from hcaptcha_challenger.components.image_downloader import Cirilla
from hcaptcha_challenger.components.prompt_handler import split_prompt_message, label_cleaning
from hcaptcha_challenger.onnx.modelhub import ModelHub, DEFAULT_KEYPOINT_MODEL
from hcaptcha_challenger.onnx.resnet import ResNetControl
from hcaptcha_challenger.onnx.yolo import YOLOv8, is_matched_ash_of_war, finetune_keypoint
from hcaptcha_challenger.utils import from_dict_to_model


@dataclass
class Status:
    # <success> Challenge Passed by following the expected
    CHALLENGE_SUCCESS = "success"
    # <continue> Continue the challenge
    CHALLENGE_CONTINUE = "continue"
    # <crash> Failure of the challenge as expected
    CHALLENGE_CRASH = "crash"
    # <retry> Your proxy IP may have been flagged
    CHALLENGE_RETRY = "retry"
    # <refresh> Skip the specified label as expected
    CHALLENGE_REFRESH = "refresh"
    # <backcall> (New Challenge) Types of challenges not yet scheduled
    CHALLENGE_BACKCALL = "backcall"
    # <to-X> NOT MATCH PATTERN
    CHALLENGE_TO_BINARY = "to_binary"
    CHALLENGE_TO_AREA_SELECT = "to_area_select"

    AUTH_SUCCESS = "success"
    AUTH_ERROR = "error"
    AUTH_CHALLENGE = "challenge"


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

    requester_restricted_answer_set: Dict[str, Any] = field(default_factory=dict)
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
    def from_json(cls, data: Dict[str, Any]):
        return from_dict_to_model(cls, data)

    def save_example(self, tmp_dir: Path = None):
        shape_type = self.request_config.get("shape_type", "")

        requester_question = label_cleaning(self.requester_question.get("en", ""))
        answer_keys = list(self.requester_restricted_answer_set.keys())
        ak = f".{answer_keys[0]}" if len(answer_keys) > 0 else ""
        fn = f"{self.request_type}.{shape_type}.{requester_question}{ak}.json"
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
    def from_json(cls, metadata: Dict[str, Any]):
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

    qr_queue: asyncio.Queue[QuestionResp] | None = None
    cr_queue: asyncio.Queue[ChallengeResp] | None = None

    this_dir: Path = Path(__file__).parent
    """
    Project directory of Skeleton Agents
    """

    tmp_dir: Path = this_dir.joinpath("tmp_dir")
    challenge_dir: Path = field(default=Path)
    record_json_dir: Path = field(default=Path)
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
        self.challenge_dir = self.tmp_dir.joinpath("_challenge")
        self.record_json_dir = self.tmp_dir.joinpath("record_json")
        self.record_json_dir.mkdir(parents=True, exist_ok=True)

        self.label_alias = self.modelhub.label_alias

        self.qr_queue = asyncio.Queue()
        self.cr_queue = asyncio.Queue()

        self.handle_question_resp(self.page)

    async def handler(self, response: Response):
        if response.url.startswith("https://hcaptcha.com/getcaptcha/"):
            with suppress(Exception):
                data = await response.json()
                qr = QuestionResp.from_json(data)
                qr.save_example(tmp_dir=self.record_json_dir)
                self.qr_queue.put_nowait(qr)
                if data.get("pass"):
                    cr = ChallengeResp.from_json(data)
                    self.cr_queue.put_nowait(cr)
        if response.url.startswith("https://hcaptcha.com/checkcaptcha/"):
            with suppress(Exception):
                metadata = await response.json()
                cr = ChallengeResp.from_json(metadata)
                self.cr_queue.put_nowait(cr)

    def handle_question_resp(self, page: Page):
        page.on("response", self.handler)

    @classmethod
    def from_page(cls, page: Page, tmp_dir=None, **kwargs):
        modelhub = ModelHub.from_github_repo(**kwargs)
        modelhub.parse_objects()

        if tmp_dir and isinstance(tmp_dir, Path):
            return cls(page=page, modelhub=modelhub, tmp_dir=tmp_dir)
        return cls(page=page, modelhub=modelhub)

    @property
    def status(self):
        return Status

    @property
    def ash(self):
        answer_keys = list(self.qr.requester_restricted_answer_set.keys())
        ak = answer_keys[0] if len(answer_keys) > 0 else ""
        ash = f"{self._label} {ak}"
        return ash

    def _switch_to_challenge_frame(self, page: Page, window: str = "login", **kwargs):
        if window == "login":
            frame_challenge = page.frame_locator(self.HOOK_CHALLENGE)
        else:
            frame_purchase = page.frame_locator(self.HOOK_PURCHASE)
            frame_challenge = frame_purchase.frame_locator(self.HOOK_CHALLENGE)

        return frame_challenge

    async def _reset_state(self):
        self.cr = None
        self.qr = await self.qr_queue.get()

    def _recover_state(self):
        if not self.cr_queue.empty():
            cr = self.cr_queue.get_nowait()
            if cr.is_pass:
                self.cr = cr

    def _parse_label(self):
        self._prompt = self.qr.requester_question.get("en")
        _label = split_prompt_message(self._prompt, lang="en")
        self._label = label_cleaning(_label)

    async def _download_images(self):
        request_type = self.qr.request_type
        ks = list(self.qr.requester_restricted_answer_set.keys())
        if len(ks) > 0:
            self.typed_dir = self.tmp_dir.joinpath(request_type, self._label, ks[0])
        else:
            self.typed_dir = self.tmp_dir.joinpath(request_type, self._label)
        self.typed_dir.mkdir(parents=True, exist_ok=True)

        ciri = Cirilla()
        container = []
        tasks = []
        for i, tk in enumerate(self.qr.tasklist):
            challenge_img_path = self.typed_dir.joinpath(f"{time.time()}.{i}.png")
            context = (challenge_img_path, tk["datapoint_uri"])
            container.append(context)
            tasks.append(asyncio.create_task(ciri.elder_blood(context)))

        await asyncio.gather(*tasks)

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
            focus_name, classes = self.modelhub.apply_ash_of_war(ash=self.ash)
            logger.debug("match model", yolo=focus_name, prompt=self._prompt)
            session = self.modelhub.match_net(focus_name=focus_name)
            detector = YOLOv8.from_pluggable_model(session, classes)
            return detector

        # Match ResNet model
        focus_name = focus_label
        if not focus_name.endswith(".onnx"):
            focus_name = f"{focus_name}.onnx"
            logger.debug("match model", resnet=focus_name, prompt=self._prompt)
        net = self.modelhub.match_net(focus_name=focus_name)
        control = ResNetControl.from_pluggable_model(net)
        return control

    async def _bounding_challenge(self, frame_challenge: FrameLocator):
        detector: YOLOv8 = self._match_solution(select="yolo")
        times = int(len(self.qr.tasklist))
        for pth in range(times):
            locator = frame_challenge.locator("//div[@class='challenge-view']//canvas")
            await locator.wait_for(state="visible")

            path = self.tmp_dir.joinpath("_challenge", f"{uuid.uuid4()}.png")
            await locator.screenshot(path=path, type="png")

            res = detector(Path(path), shape_type="bounding_box")

            alts = []
            for name, (x1, y1), (x2, y2), score in res:
                if not is_matched_ash_of_war(ash=self.ash, class_name=name):
                    continue
                scoop = (x2 - x1) * (y2 - y1)
                start = (int(x1), int(y1))
                end = (int(x2), int(y2))
                alt = {"name": name, "start": start, "end": end, "scoop": scoop}
                alts.append(alt)

            if len(alts) > 1:
                alts = sorted(alts, key=lambda xf: xf["scoop"])
            if len(alts) > 0:
                best = alts[-1]
                x1, y1 = best["start"]
                x2, y2 = best["end"]
                await locator.click(delay=200, position={"x": x1, "y": y1})
                await self.page.mouse.move(x2, y2)
                await locator.click(delay=200, position={"x": x2, "y": y2})

            with suppress(TimeoutError):
                fl = frame_challenge.locator("//div[@class='button-submit button']")
                await fl.click(delay=200)

            if pth == 0:
                await self.page.wait_for_timeout(1000)

    async def _keypoint_default_challenge(self, frame_challenge: FrameLocator):
        def lookup_objects(deep: int = 6) -> Position[str, str] | None:
            count = 0
            for focus_name, classes in self.modelhub.lookup_ash_of_war(self.ash):
                count += 1
                session = self.modelhub.match_net(focus_name=focus_name)
                detector = YOLOv8.from_pluggable_model(session, classes)
                res = detector(image, shape_type="point")
                self.modelhub.unplug()
                for name, (center_x, center_y), score in res:
                    if center_y < 20 or center_y > 520 or center_x < 91 or center_x > 400:
                        continue
                    logger.debug("catch model", yolo=focus_name, ash=self.ash)
                    return {"x": center_x, "y": center_y}
                if count > deep:
                    return

        def lookup_unique_object() -> Position[int, int] | None:
            classes = self.modelhub.ashes_of_war.get(DEFAULT_KEYPOINT_MODEL)
            session = self.modelhub.match_net(focus_name=DEFAULT_KEYPOINT_MODEL)
            detector = YOLOv8.from_pluggable_model(session, classes)
            results = detector(path, shape_type="point")
            self.modelhub.unplug()
            logger.debug("select model", yolo=DEFAULT_KEYPOINT_MODEL, ash=self.ash)
            img, circles = annotate_objects(str(path))
            if results:
                circles = [[int(result[1][0]), int(result[1][1]), 32] for result in results]
            if circles:
                if result := find_unique_object(img, circles):
                    x, y, _ = result
                    return {"x": int(x), "y": int(y)}

        times = int(len(self.qr.tasklist))
        for pth in range(times):
            locator = frame_challenge.locator("//div[@class='challenge-view']//canvas")
            await locator.wait_for(state="visible")
            await self.page.wait_for_timeout(800)

            path = self.tmp_dir.joinpath("_challenge", f"{uuid.uuid4()}.png")
            image = await locator.screenshot(path=path, type="png")

            if "appears only once" in self.ash or "never repeated" in self.ash:
                position = lookup_unique_object()
            else:
                position = lookup_objects()

            if position:
                await locator.click(delay=500, position=position)
            else:
                await locator.click(delay=500)

            # {{< Verify >}}
            with suppress(TimeoutError):
                fl = frame_challenge.locator("//div[@class='button-submit button']")
                await fl.click(delay=200)

            # {{< Done | Continue >}}
            if pth == 0:
                await self.page.wait_for_timeout(1000)

    async def _keypoint_challenge(self, frame_challenge: FrameLocator):
        # Load YOLOv8 model from local or remote repo
        detector: YOLOv8 = self._match_solution(select="yolo")

        # Execute the detection task for twice
        times = int(len(self.qr.tasklist))
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
                if not is_matched_ash_of_war(ash=self.ash, class_name=name):
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
        times = int(len(self.qr.tasklist) / 9)
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
                fl = frame_challenge.locator("//div[@class='button-submit button']")
                await fl.click()

            # {{< Done | Continue >}}
            if pth == 0:
                await self.page.wait_for_timeout(300)

    async def _is_success(self):
        self.cr = await self.cr_queue.get()
        if not self.cr or not self.cr.is_pass:
            return self.status.CHALLENGE_RETRY
        if self.cr.is_pass:
            return self.status.CHALLENGE_SUCCESS


@dataclass
class AgentT(Radagon):
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
        if not self.cr:
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

        _rqdata_path.write_text(json.dumps(self.cr.__dict__, indent=2))

        return _rqdata_path

    async def handle_checkbox(self):
        with suppress(TimeoutError):
            checkbox = self.page.frame_locator("//iframe[contains(@title,'checkbox')]")
            await checkbox.locator("#checkbox").click()

    async def execute(self, **kwargs) -> str | None:
        window = kwargs.get("window", "login")

        frame_challenge = self._switch_to_challenge_frame(self.page, window)

        await self._reset_state()

        # Match: ChallengePassed
        if not self.qr.requester_question.keys():
            self._recover_state()
            return self.status.CHALLENGE_SUCCESS

        self._parse_label()

        await self._download_images()

        # Match: image_label_binary
        if self.qr.request_type == "image_label_binary":
            if not self.label_alias.get(self._label):
                return self.status.CHALLENGE_BACKCALL
            await self._binary_challenge(frame_challenge)
        # Match: image_label_area_select
        elif self.qr.request_type == "image_label_area_select":
            ash = self.ash
            shape_type = self.qr.request_config.get("shape_type", "")

            if "default" in ash:
                if shape_type == "point":
                    await self._keypoint_default_challenge(frame_challenge)
                else:
                    return self.status.CHALLENGE_BACKCALL
            else:
                if not any(is_matched_ash_of_war(ash, c) for c in self.modelhub.yolo_names):
                    return self.status.CHALLENGE_BACKCALL
                if shape_type == "point":
                    await self._keypoint_challenge(frame_challenge)
                elif shape_type == "bounding_box":
                    await self._bounding_challenge(frame_challenge)

        self.modelhub.unplug()

        result = await self._is_success()
        return result

    async def collect(self) -> str | None:
        """Download datasets"""
        await self._reset_state()
        if not self.qr.requester_question.keys():
            return
        self._parse_label()
        await self._download_images()
        return self._label
