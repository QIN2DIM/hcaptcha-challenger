# -*- coding: utf-8 -*-
# Time       : 2023/8/25 14:05
# Author     : QIN2DIM
# GitHub     : https://github.com/QIN2DIM
# Description:
from __future__ import annotations

import asyncio
import random
import time
import uuid
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Literal, Iterable

from PIL import Image
from loguru import logger
from playwright.async_api import Page, FrameLocator, Response, Position, Locator
from playwright.async_api import TimeoutError
from tenacity import *

from hcaptcha_challenger.components.common import (
    match_model,
    match_datalake,
    rank_models,
    download_challenge_images,
)
from hcaptcha_challenger.components.cv_toolkit import (
    find_unique_object,
    annotate_objects,
    find_unique_color,
)
from hcaptcha_challenger.components.middleware import (
    Status,
    QuestionResp,
    ChallengeResp,
    RequestType,
)
from hcaptcha_challenger.components.prompt_handler import handle
from hcaptcha_challenger.components.zero_shot_image_classifier import (
    ZeroShotImageClassifier,
    register_pipline,
)
from hcaptcha_challenger.onnx.modelhub import ModelHub, DataLake
from hcaptcha_challenger.onnx.resnet import ResNetControl
from hcaptcha_challenger.onnx.yolo import (
    YOLOv8,
    YOLOv8Seg,
    is_matched_ash_of_war,
    finetune_keypoint,
)


@dataclass
class Radagon:
    HOOK_PURCHASE = "//div[@id='webPurchaseContainer']//iframe"
    HOOK_CHECKBOX = "//iframe[contains(@title, 'checkbox for hCaptcha')]"
    HOOK_CHALLENGE = "//iframe[contains(@title, 'hCaptcha challenge')]"

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

    img_paths: List[Path] = field(default_factory=list)
    """
    bytes of challenge image
    """
    example_paths: List[Path] = field(default_factory=list)
    """
    bytes of example image
    """

    label = ""
    """
    Cleaned Challenge Prompt in the context
    """

    prompt = ""
    """
    Challenge Prompt in the context
    """

    label_alias: Dict[str, str] = field(default_factory=dict)
    """
    A collection of { prompt[s]: model_name[.onnx] }
    """

    nested_categories: Dict[str, List[str]] = field(default_factory=dict)

    self_supervised: bool = True

    def __post_init__(self):
        self.challenge_dir = self.tmp_dir.joinpath("_challenge")
        self.record_json_dir = self.tmp_dir.joinpath("record_json")
        self.record_json_dir.mkdir(parents=True, exist_ok=True)

        self.label_alias = self.modelhub.label_alias
        self.nested_categories = self.modelhub.nested_categories

        self.qr_queue = asyncio.Queue()
        self.cr_queue = asyncio.Queue()

        self.handle_question_resp(self.page)

    async def handler(self, response: Response):
        if response.url.startswith("https://api.hcaptcha.com/getcaptcha/"):
            try:
                data = await response.json()
                qr = QuestionResp(**data)
                qr.cache(tmp_dir=self.record_json_dir)
                self.qr_queue.put_nowait(qr)
                if data.get("pass"):
                    cr = ChallengeResp(**data)
                    self.cr_queue.put_nowait(cr)
            except Exception as err:
                logger.exception(err)
        if response.url.startswith("https://api.hcaptcha.com/checkcaptcha/"):
            try:
                metadata = await response.json()
                cr = ChallengeResp(**metadata)
                self.cr_queue.put_nowait(cr)
            except Exception as err:
                logger.exception(err)

    def handle_question_resp(self, page: Page):
        page.on("response", self.handler)

    @classmethod
    def from_page(
        cls,
        page: Page,
        tmp_dir=None,
        modelhub: ModelHub | None = None,
        self_supervised: bool | None = True,
        **kwargs,
    ):
        modelhub = modelhub or ModelHub.from_github_repo(**kwargs)
        if not modelhub.label_alias:
            modelhub.parse_objects()

        if not isinstance(tmp_dir, Path):
            tmp_dir = Path(__file__).parent.parent.joinpath("tmp_dir")

        self_supervised = kwargs.get("clip", self_supervised)

        return cls(page=page, modelhub=modelhub, tmp_dir=tmp_dir, self_supervised=self_supervised)

    @property
    def status(self):
        return Status

    @property
    def ash(self):
        answer_keys = list(self.qr.requester_restricted_answer_set.keys())
        ak = answer_keys[0] if len(answer_keys) > 0 else ""
        ash = f"{self.label} {ak}"
        return ash

    def _switch_to_challenge_frame(self, page: Page, window: str = "login", **kwargs):
        if window == "login":
            frame_challenge = page.frame_locator(self.HOOK_CHALLENGE)
        else:
            frame_purchase = page.frame_locator(self.HOOK_PURCHASE)
            frame_challenge = frame_purchase.frame_locator(self.HOOK_CHALLENGE)

        return frame_challenge

    @retry(
        retry=retry_if_exception_type(asyncio.QueueEmpty),
        wait=wait_fixed(0.5),
        stop=(stop_after_delay(30) | stop_after_attempt(60)),
        reraise=True,
    )
    async def _reset_state(self) -> bool | None:
        self.cr = None
        self.qr = self.qr_queue.get_nowait()

        return True

    @retry(
        retry=retry_if_exception_type(asyncio.QueueEmpty),
        wait=wait_random_exponential(multiplier=1, max=60),
        stop=(stop_after_delay(30) | stop_after_attempt(15)),
        reraise=True,
    )
    async def _is_success(self):
        self.cr = self.cr_queue.get_nowait()

        # Match: Timeout / Loss
        if not self.cr or not self.cr.is_pass:
            return self.status.CHALLENGE_RETRY
        if self.cr.is_pass:
            return self.status.CHALLENGE_SUCCESS

    def _recover_state(self):
        if not self.cr_queue.empty():
            cr = self.cr_queue.get_nowait()
            if cr.is_pass:
                self.cr = cr

    def _parse_label(self):
        self.prompt = self.qr.requester_question.get("en")
        self.label = handle(self.prompt)
        logger.debug("get task", prompt=self.prompt)

    async def _download_images(self, *, ignore_examples: bool = False):
        self.img_paths, self.example_paths = await download_challenge_images(
            self.qr, self.label, self.tmp_dir, ignore_examples=ignore_examples
        )

    def _match_model(self, select: Literal["yolo", "resnet"] = None) -> ResNetControl | YOLOv8:
        """match solution after `tactical_retreat`"""
        model = match_model(self.label, self.ash, self.modelhub, select=select)
        logger.debug("handle task", match_model=select, prompt=self.prompt)

        return model

    def _rank_models(self, nested_models: List[str]) -> ResNetControl | None:
        result = rank_models(nested_models, self.example_paths, self.modelhub)
        if result and isinstance(result, tuple):
            best_model, model_name = result
            logger.debug("handle task", rank_model=model_name, prompt=self.prompt)
            return best_model

    async def _bounding_challenge(self, frame_challenge: FrameLocator):
        detector: YOLOv8 = self._match_model(select="yolo")
        times = len(self.qr.tasklist)
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
        def lookup_objects(_iter_launcher: Iterable, deep: int = 6) -> Position[str, str] | None:
            count = 0
            for focus_name, classes in _iter_launcher:
                count += 1
                session = self.modelhub.match_net(focus_name=focus_name)
                if "-seg" in focus_name:
                    detector = YOLOv8Seg.from_pluggable_model(session, classes)
                    res = detector(path, shape_type="point")
                else:
                    detector = YOLOv8.from_pluggable_model(session, classes)
                    res = detector(image, shape_type="point")
                self.modelhub.unplug()
                for name, (center_x, center_y), score in res:
                    if center_y < 20 or center_y > 520 or center_x < 91 or center_x > 400:
                        continue
                    logger.debug("handle task", catch_model=focus_name, ash=self.ash)
                    return {"x": center_x, "y": center_y}
                if count > deep:
                    return

        def lookup_unique_object(trident) -> Position[int, int] | None:
            model_name = self.modelhub.circle_segment_model
            classes = self.modelhub.ashes_of_war.get(model_name)
            session = self.modelhub.match_net(model_name)
            detector = YOLOv8Seg.from_pluggable_model(session, classes)
            results = detector(path, shape_type="point")
            self.modelhub.unplug()
            img, circles = annotate_objects(str(path))
            # Extract point coordinates
            if results:
                circles = [[int(result[1][0]), int(result[1][1]), 32] for result in results]
                logger.debug(
                    "handle task", select_model=model_name, trident=trident.__name__, ash=self.ash
                )
            # Filter points outside the bounding box
            edge_circles = []
            if len(self.example_paths) == 0:
                edge_circles = circles
            else:
                for circle in circles:
                    x, y, r = circle
                    if y < 20 or y > 520 or x < 91 or x > 400:
                        continue
                    edge_circles.append([x, y, r])
            # Find targets with special semantics
            if edge_circles:
                if result := trident(img, edge_circles):
                    x, y, _ = result
                    return {"x": int(x), "y": int(y)}

        times = len(self.qr.tasklist)
        for pth in range(times):
            locator = frame_challenge.locator("//div[@class='challenge-view']//canvas")
            await locator.wait_for(state="visible")
            await self.page.wait_for_timeout(800)

            path = self.tmp_dir.joinpath("_challenge", f"{uuid.uuid4()}.png")
            image = await locator.screenshot(path=path, type="png")

            position = None
            launcher = []

            if nested_models := self.nested_categories.get(self.label, []):
                for model_name in nested_models:
                    element = model_name, self.modelhub.ashes_of_war.get(model_name, [])
                    launcher.append(element)
                if launcher:
                    position = lookup_objects(launcher)
            elif "appears only once" in self.ash or "never repeated" in self.ash:
                position = lookup_unique_object(trident=find_unique_object)
            elif "shapes are of the same color" in self.ash:
                position = lookup_unique_object(trident=find_unique_color)
            else:
                launcher = self.modelhub.lookup_ash_of_war(self.ash)
                position = lookup_objects(launcher)

            await self.page.wait_for_timeout(800)
            if position:
                await locator.click(position=position)
            else:
                await locator.click()

            # {{< Verify >}}
            with suppress(TimeoutError):
                fl = frame_challenge.locator("//div[@class='button-submit button']")
                await fl.click(delay=200)

    async def _keypoint_challenge(self, frame_challenge: FrameLocator):
        # Load YOLOv8 model from local or remote repo
        detector: YOLOv8 = self._match_model(select="yolo")

        # Execute the detection task for twice
        times = len(self.qr.tasklist)
        for pth in range(times):
            locator = frame_challenge.locator("//div[@class='challenge-view']//canvas")
            await locator.wait_for(state="visible")

            path = self.tmp_dir.joinpath("_challenge", f"{uuid.uuid4()}.png")
            await locator.screenshot(path=path, type="png")

            # {{< Please click on the X >}}
            res = detector(Path(path), shape_type="point")

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
                await locator.click(delay=200, position=best["position"])
            # Catch-all rule
            else:
                await locator.click(delay=200)

            # {{< Verify >}}
            with suppress(TimeoutError):
                fl = frame_challenge.locator("//div[@class='button-submit button']")
                await fl.click(delay=200)

            # {{< Done | Continue >}}
            if pth == 0:
                await self.page.wait_for_timeout(1000)

    async def _binary_challenge(self, frame_challenge: FrameLocator, model: ResNetControl = None):
        classifier = model or self._match_model(select="resnet")

        # {{< IMAGE CLASSIFICATION >}}
        times = int(len(self.qr.tasklist) / 9)
        for pth in range(times):
            # Drop element location
            samples = frame_challenge.locator("//div[@class='task-image']")
            count = await samples.count()
            # Classify and Click on the right image
            positive_cases = 0
            for i in range(count):
                sample = samples.nth(i)
                await sample.wait_for()
                result = classifier.execute(img_stream=self.img_paths[i + pth * 9].read_bytes())
                if result:
                    positive_cases += 1
                    with suppress(TimeoutError):
                        await sample.click(delay=200)
                elif positive_cases == 0 and pth == times - 1 and i == count - 1:
                    await sample.click(delay=200)

            # {{< Verify >}}
            await asyncio.sleep(random.uniform(0.1, 0.3))
            with suppress(TimeoutError):
                fl = frame_challenge.locator("//div[@class='button-submit button']")
                await fl.click()

    async def _catch_all_binary_challenge(self, frame_challenge: FrameLocator):
        dl = match_datalake(self.modelhub, self.label)
        tool = ZeroShotImageClassifier.from_datalake(dl)

        # Default to `RESNET.OPENAI` perf_counter 1.794s
        t0 = time.perf_counter()
        model = register_pipline(self.modelhub)
        te = time.perf_counter()

        logger.debug(
            "handle task",
            unsupervised="binary",
            candidate_labels=tool.candidate_labels,
            prompt=self.prompt,
            timit=f"{te - t0:.3f}s",
        )

        # {{< CATCH EXAMPLES >}}
        target = {}
        if self.example_paths:
            example_path = self.example_paths[-1]
            results = tool(model, image=Image.open(example_path))
            target = results[0]

        # {{< IMAGE CLASSIFICATION >}}
        times = int(len(self.qr.tasklist) / 9)
        for pth in range(times):
            samples = frame_challenge.locator("//div[@class='task-image']")
            count = await samples.count()
            positive_cases = 0
            for i in range(count):
                sample = samples.nth(i)
                await sample.wait_for()
                results = tool(model, image=Image.open(self.img_paths[i + pth * 9]))

                if (
                    results[0]["label"] in target.get("label", "")
                    or results[0]["label"] in tool.positive_labels
                ):
                    positive_cases += 1
                    with suppress(TimeoutError):
                        await sample.click(delay=200)
                elif positive_cases == 0 and pth == times - 1 and i == count - 1:
                    await sample.click(delay=200)

            # {{< Verify >}}
            with suppress(TimeoutError):
                fl = frame_challenge.locator("//div[@class='button-submit button']")
                await fl.click()

    async def _multiple_choice_challenge(self, frame_challenge: FrameLocator):
        def inject_datalake(img_path: Path) -> Locator | None:
            candidates = [lb["text"] for lb in label_btn]
            dl = DataLake.from_binary_labels(
                positive_labels=candidates[:1], negative_labels=candidates[1:]
            )
            tool = ZeroShotImageClassifier.from_datalake(dl)
            results = tool(model, image=Image.open(img_path))
            sample_label = results[0]["label"]
            logger.debug(
                "handle task",
                unsupervised="multiple choice",
                results=sample_label,
                candidate_labels=candidates,
                prompt=self.prompt,
            )
            for lb in label_btn:
                if DataLake.PREMISED_YES.format(lb["text"]) == sample_label:
                    return lb["btn"]

        model = register_pipline(self.modelhub)

        times = len(self.qr.tasklist)
        for pth in range(times):
            await self.page.wait_for_timeout(300)
            label_btn: List[Dict[str, str | Locator]] = []
            samples = frame_challenge.locator("//div[@class='challenge-answer']")
            count = await samples.count()
            for i in range(count):
                sample = samples.nth(i)
                await sample.wait_for()
                text_content = await sample.text_content()
                label_btn.append({"text": text_content.strip(), "btn": sample})

            if btn := inject_datalake(img_path=self.img_paths[pth]):
                await btn.click(delay=200)
            else:
                await label_btn[-1]["btn"].click(delay=200)

            # {{< Verify >}}
            with suppress(TimeoutError):
                fl = frame_challenge.locator("//div[@class='button-submit button']")
                await fl.click()


@dataclass
class AgentT(Radagon):
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

        _rqdata_path.write_text(self.cr.model_dump_json(indent=2), encoding="utf8")

        return _rqdata_path

    async def handle_checkbox(self):
        with suppress(TimeoutError):
            checkbox = self.page.frame_locator("//iframe[contains(@title,'checkbox')]")
            await checkbox.locator("#checkbox").click()

    async def execute(self, **kwargs) -> Status | None:
        window = kwargs.get("window", "login")

        frame_challenge = self._switch_to_challenge_frame(self.page, window)

        # Match: Failed to obtain challenge task
        try:
            if not await self._reset_state() or not self.qr:
                raise asyncio.QueueEmpty
        except asyncio.QueueEmpty:
            logger.error(
                f"task interrupt <{self.status.CHALLENGE_BACKCALL}>",
                reason="Failed to obtain challenge task",
            )
            return self.status.CHALLENGE_BACKCALL

        # Match: ChallengePassed
        if not self.qr.requester_question.keys():
            self._recover_state()
            return self.status.CHALLENGE_SUCCESS

        self._parse_label()

        await self._download_images()

        # Match: image_label_binary
        if self.qr.request_type == RequestType.ImageLabelBinary:
            if nested_models := self.nested_categories.get(self.label, []):
                if model := self._rank_models(nested_models):
                    await self._binary_challenge(frame_challenge, model)
                elif self.self_supervised:
                    await self._catch_all_binary_challenge(frame_challenge)
                else:
                    return self.status.CHALLENGE_BACKCALL
            elif self.label_alias.get(self.label):
                await self._binary_challenge(frame_challenge)
            elif self.self_supervised:
                await self._catch_all_binary_challenge(frame_challenge)
            else:
                return self.status.CHALLENGE_BACKCALL
        # Match: image_label_area_select
        elif self.qr.request_type == RequestType.ImageLabelAreaSelect:
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
        # Match: image_label_multiple_choice
        elif self.qr.request_type == RequestType.ImageLabelMultipleChoice:
            # By default, CLIP is used to process this type of task.
            if not self.self_supervised:
                return self.status.CHALLENGE_BACKCALL
            await self._multiple_choice_challenge(frame_challenge)
        # Match: Unknown case
        else:
            logger.warning("task interrupt", reason="Unknown type of challenge")
            return self.status.CHALLENGE_BACKCALL

        self.modelhub.unplug()

        # Match: Failed to obtain challenge response
        try:
            result = await self._is_success()
            return result
        except asyncio.QueueEmpty:
            logger.error(
                f"task interrupt <{self.status.CHALLENGE_BACKCALL}>",
                reason="Failed to obtain challenge response",
            )
            return self.status.CHALLENGE_BACKCALL

    async def collect(self) -> str | None:
        """Download datasets"""
        await self._reset_state()
        if not self.qr.requester_question.keys():
            return
        self._parse_label()
        await self._download_images(ignore_examples=True)
        return self.label
