# -*- coding: utf-8 -*-
# Time       : 2024/4/7 11:43
# Author     : QIN2DIM
# GitHub     : https://github.com/QIN2DIM
# Description:
import abc
import asyncio
import hashlib
import os
import re
import uuid
from abc import ABC
from asyncio import Queue
from contextlib import suppress
from dataclasses import dataclass
from dataclasses import field
from enum import Enum
from pathlib import Path
from typing import List

import dotenv
import httpx
from asyncache import cached
from cachetools import TTLCache
from loguru import logger
from playwright.async_api import Page, Response, TimeoutError, expect

from hcaptcha_challenger.constant import INV
from hcaptcha_challenger.helper import inject_mouse_visualizer_global
from hcaptcha_challenger.models import (
    ChallengeResp,
    QuestionResp,
    RequestType,
    ChallengeImage,
    ToolExecution,
    CollectibleType,
    Collectible,
    SelfSupervisedPayload,
)
from hcaptcha_challenger.onnx.modelhub import ModelHub
from hcaptcha_challenger.tools import handle, invoke_clip_tool

dotenv.load_dotenv()

HOOK_PURCHASE = "//div[@id='webPurchaseContainer']//iframe"
HOOK_CHECKBOX = "//iframe[contains(@title, 'checkbox for hCaptcha')]"
HOOK_CHALLENGE = "//iframe[contains(@title, 'hCaptcha challenge')]"

_cached_ping_result = TTLCache(maxsize=10, ttl=60)


class ChallengeSignal(str, Enum):
    """
    Represents the possible statuses of a challenge.

    Enum Members:
      SUCCESS: The challenge was completed successfully.
      FAILURE: The challenge failed or encountered an error.
      START: The challenge has been initiated or started.
    """

    SUCCESS = "success"
    FAILURE = "failure"
    START = "start"
    TIMEOUT = "timeout"
    RETRY = "retry"
    QR_DATA_NOT_FOUND = "qr_data_not_found"
    EXECUTION_TIMEOUT = "challenge_execution_timeout"
    RESPONSE_TIMEOUT = "challenge_response_timeout"


@cached(_cached_ping_result)
async def is_solver_edge_worker_available() -> bool:
    solver_base_url = os.getenv("SOLVER_BASE_URL")
    if not solver_base_url:
        return False

    try:
        client = httpx.AsyncClient(base_url=solver_base_url, timeout=1)
        response = await client.get("/ping")
        response.raise_for_status()
        return True
    except (httpx.HTTPStatusError, httpx.ReadTimeout) as err:
        logger.warning("Failed to connect SolverEdgeWorker", base_url=solver_base_url, err=err)
        return False


@dataclass
class SolverEdgeWorker:
    solver_base_url: str | None = os.getenv("SOLVER_BASE_URL")
    """
    Default to http://localhost:33777
    """

    def __init__(self):
        self.client = None
        if self.solver_base_url:
            self.client = httpx.AsyncClient(base_url=self.solver_base_url)

    async def invoke_clip_tool(self, payload: dict) -> List[bool]:
        _service_point = "/challenge/image_label_binary"

        response = await self.client.post(_service_point, json=payload)
        response.raise_for_status()
        results = response.json()["results"]

        return results


class RoboticArm:

    def __init__(self, page: Page):
        self.page = page
        self.sew = SolverEdgeWorker()

        self.modelhub = ModelHub.from_github_repo()
        self.modelhub.parse_objects()

    @property
    def checkbox_selector(self) -> str:
        return "//iframe[starts-with(@src,'https://newassets.hcaptcha.com/captcha/v1/') and contains(@src, 'frame=checkbox')]"

    @property
    def challenge_selector(self) -> str:
        return "//iframe[starts-with(@src,'https://newassets.hcaptcha.com/captcha/v1/') and contains(@src, 'frame=challenge')]"

    async def click_checkbox(self):
        checkbox_frame = self.page.frame_locator(self.checkbox_selector)

        # 获取checkbox元素
        checkbox_element = checkbox_frame.locator("//div[@id='checkbox']")

        # 获取元素的位置信息
        bbox = await checkbox_element.bounding_box()

        # 计算元素中心点坐标
        center_x = bbox['x'] + bbox['width'] / 2
        center_y = bbox['y'] + bbox['height'] / 2

        # 移动鼠标到元素位置
        await self.page.mouse.move(center_x, center_y)

        # 点击鼠标左键
        await self.page.mouse.click(center_x, center_y, delay=150)

    async def refresh_challenge(self) -> bool | None:
        try:
            fl = self.page.frame_locator(HOOK_CHALLENGE)
            await fl.locator("//div[@class='refresh button']").click()
            return True
        except TimeoutError as err:
            logger.warning("Failed to click refresh button", reason=err)

    def switch_to_challenge_frame(self, window: str = "login"):
        if window == "login":
            frame_challenge = self.page.frame_locator(HOOK_CHALLENGE)
        else:
            frame_purchase = self.page.frame_locator(HOOK_PURCHASE)
            frame_challenge = frame_purchase.frame_locator(HOOK_CHALLENGE)

        return frame_challenge

    async def challenge_image_label_binary(
        self, label: str, challenge_images: List[ChallengeImage]
    ):
        frame_challenge = self.switch_to_challenge_frame()

        # {{< Reload SELF-SUPERVISED CONFIGURATION >}}
        if not (model_slot := self.modelhub.model_slots.get(label)):
            return
        if not (clip_selection := model_slot.clip_selection):
            return

        challenge_images = [i.into_base64bytes() for i in challenge_images]
        clip_selection = clip_selection.model_dump()
        self_supervised_payload = {
            "prompt": label,
            "challenge_images": challenge_images,
            **clip_selection,
        }

        # {{< IMAGE CLASSIFICATION >}}
        if await is_solver_edge_worker_available():
            results: List[bool] = await self.sew.invoke_clip_tool(self_supervised_payload)
        else:
            payload = SelfSupervisedPayload(**self_supervised_payload)
            results: List[bool] = invoke_clip_tool(self.modelhub, payload, self.clip_model)

        # {{< DRIVE THE BROWSER TO WORK ON THE CHALLENGE >}}
        samples = frame_challenge.locator("//div[@class='task-image']")
        count = await samples.count()
        positive_cases = 0
        for i in range(count):
            sample = samples.nth(i)
            if results[i]:
                positive_cases += 1
                with suppress(TimeoutError):
                    await sample.click(delay=200)
            elif positive_cases == 0 and i == count - 1:
                await sample.click(delay=200)

        # {{< Verify >}}
        with suppress(TimeoutError):
            fl = frame_challenge.locator("//div[@class='button-submit button']")
            await fl.click()


@dataclass
class OminousLand(ABC):
    """不祥之地：base"""

    page: Page
    tmp_dir: Path
    robotic_arm: RoboticArm
    image_queue: Queue

    crumb_count = 1

    typed_dir: Path = field(default_factory=Path)
    canvas_screenshot_dir: Path = field(default_factory=Path)

    label: str = ""
    prompt: str = ""

    tasklist: List[ChallengeImage] = field(default_factory=list)
    examples: List[ChallengeImage] = field(default_factory=list)

    qr_data: dict | None = field(default_factory=dict)
    qr: QuestionResp | None = field(default_factory=QuestionResp)

    encrypted_bytes: bytes | None = b""

    @classmethod
    def draws_from(
        cls,
        page: Page,
        inputs: dict | bytes,
        tmp_dir: Path,
        image_queue: Queue,
        ms: RoboticArm | None = None,
    ):
        # Cache images
        if not isinstance(tmp_dir, Path):
            tmp_dir = Path("tmp_dir")

        typed_dir = tmp_dir / "typed_dir"
        canvas_screenshot_dir = tmp_dir / "canvas_screenshot"

        ms = ms or RoboticArm(page=page)
        monster = cls(
            page=page,
            robotic_arm=ms,
            tmp_dir=tmp_dir,
            image_queue=image_queue,
            typed_dir=typed_dir,
            canvas_screenshot_dir=canvas_screenshot_dir,
        )

        if isinstance(inputs, dict):
            monster.qr_data = inputs
        elif isinstance(inputs, bytes):
            monster.encrypted_bytes = inputs

        return monster

    def _init_imgdb(self, label: str):
        """run after _get_captcha"""
        self.tasklist.clear()
        self.examples.clear()

        for c in INV:
            label = label.replace(c, "")

        self.typed_dir = self.tmp_dir.joinpath(self.qr.request_type, label)
        self.typed_dir.mkdir(parents=True, exist_ok=True)

        self.canvas_screenshot_dir = self.tmp_dir.joinpath(
            f"canvas_screenshot/{self.qr.request_type}/{label}"
        )
        self.canvas_screenshot_dir.mkdir(parents=True, exist_ok=True)

    async def _recall_crumb(self):
        frame_challenge = self.robotic_arm.switch_to_challenge_frame()
        crumbs = frame_challenge.locator("//div[@class='Crumb']")
        if await crumbs.first.is_visible():
            self.crumb_count = 2
        else:
            self.crumb_count = 1

    async def _recall_tasklist(self, capture_screenshot: bool = True):
        """run after _init_imgdb"""
        frame_challenge = self.robotic_arm.switch_to_challenge_frame()

        if self.qr.request_type == RequestType.ImageLabelBinary:
            images = frame_challenge.locator("//div[@class='task-grid']//div[@class='image']")
            count = await images.count()

            background_urls = []
            challenge_images = {}
            for i in range(count):
                image = images.nth(i)
                await expect(image).to_have_attribute(
                    "style", re.compile(r"url\(.+\)"), timeout=5000
                )
                style = await image.get_attribute("style")
                datapoint_uri = style.split('"')[1]
                background_urls.append(datapoint_uri)

            logger.debug(f"{self.image_queue.qsize()=}")
            while not self.image_queue.empty():
                challenge_image: ChallengeImage = self.image_queue.get_nowait()
                challenge_image.move_to(self.typed_dir)
                challenge_images[challenge_image.datapoint_uri] = challenge_image

            for url in background_urls:
                challenge_image: ChallengeImage = challenge_images.get(url)
                if challenge_image:
                    self.tasklist.append(challenge_image)

            if capture_screenshot:
                canvas = frame_challenge.locator("//div[@class='challenge-container']")
                fp = self.canvas_screenshot_dir / f"{uuid.uuid4()}.png"
                await canvas.screenshot(type="png", path=fp, scale="css")

        elif self.qr.request_type == RequestType.ImageLabelAreaSelect:
            # For the object detection task, tasklist is only used to collect datasets.
            # The challenge in progress uses a canvas screenshot, not the challenge-image
            canvas_bgk = frame_challenge.locator("//div[class='bounding-box-example']")
            await expect(canvas_bgk).not_to_be_attached()

            # Expect only 1 image in the image_queue
            while not self.image_queue.empty():
                challenge_image: ChallengeImage = self.image_queue.get_nowait()
                challenge_image.move_to(self.typed_dir)
                self.tasklist.append(challenge_image)

                if self.image_queue.qsize() == 0:
                    canvas = frame_challenge.locator("//canvas")
                    fp = self.canvas_screenshot_dir / f"{challenge_image.filename}.png"
                    await canvas.screenshot(type="png", path=fp, scale="css")

    @abc.abstractmethod
    async def _get_captcha(self, **kwargs):
        raise NotImplementedError

    async def _solve_captcha(self):
        match self.qr.request_type:
            case RequestType.ImageLabelBinary:
                try:
                    await self.robotic_arm.challenge_image_label_binary(
                        label=self.label, challenge_images=self.tasklist
                    )
                except Exception as err:
                    logger.error(f"An error occurred while processing the challenge task", err=err)
                    await self.robotic_arm.refresh_challenge()
            case RequestType.ImageLabelAreaSelect:
                await self.robotic_arm.refresh_challenge()
            case RequestType.ImageLabelMultipleChoice:
                await self.robotic_arm.refresh_challenge()
            case _:
                logger.warning("[INTERRUPT]", reason="Unknown type of challenge")

    async def _collect(self, capture_screenshot: bool = True):
        await self._get_captcha()

        logger.debug(
            "Invoke task",
            label=self.label,
            type=self.qr.request_type,
            requester_question=self.qr.requester_question,
            trigger=self.__class__.__name__,
        )

        self._init_imgdb(self.label)
        await self._recall_tasklist(capture_screenshot=capture_screenshot)

    async def _challenge(self):
        await self._collect()
        await self._recall_crumb()

        for i in range(self.crumb_count):
            if i != 0:
                await self._recall_tasklist()
            await self._solve_captcha()

    async def invoke(self, execution: ToolExecution = ToolExecution.CHALLENGE):
        match execution:
            case ToolExecution.COLLECT:
                await self._collect()
            case ToolExecution.CHALLENGE:
                await self._challenge()


@dataclass
class ChallengeProblemParser(OminousLand):
    """赤髯 (Chì Rán) ->> json"""

    async def _get_captcha(self, **kwargs):
        self.qr = QuestionResp(**self.qr_data)

        self.prompt = self.qr.requester_question.get("en")
        self.label = handle(self.prompt)


@dataclass
class ChallengeProblemDecoder(OminousLand):
    """晦月魔君 ->> bytes"""

    async def _get_captcha(self, **kwargs):
        self.qr = QuestionResp()

        # IMPORTANT
        await self.page.wait_for_timeout(2000)

        frame_challenge = self.robotic_arm.switch_to_challenge_frame()

        # requester_question
        prompt_element = frame_challenge.locator("//h2[@class='prompt-text']")
        self.prompt = await prompt_element.text_content()
        self.label = handle(self.prompt)
        lang = await frame_challenge.locator("//html").get_attribute("lang")
        self.qr.requester_question[lang] = self.prompt

        # request_type
        if await frame_challenge.locator("//div[@class='task-grid']").count():
            self.qr.request_type = RequestType.ImageLabelBinary.value
            # has_exp = await frame_challenge.locator("//div[@class='challenge-example']").count()
        elif await frame_challenge.locator("//div[contains(@class, 'bounding-box')]").count():
            self.qr.request_type = RequestType.ImageLabelAreaSelect.value
        else:
            # todo image_label_multiple_choice
            self.qr.request_type = RequestType.ImageLabelMultipleChoice.value


class AgentV:

    def __init__(self, page: Page, tmp_dir: Path = None, **kwargs):
        self.page = page

        self.robotic_arm = RoboticArm(page=page)

        self.tmp_dir = Path("tmp_dir")
        if isinstance(tmp_dir, Path):
            self.tmp_dir = tmp_dir

        self._tool_type: ToolExecution | None = None

        self._cache_dir = self.tmp_dir / ".cache"
        self._cache_dir.mkdir(parents=True, exist_ok=True)

        self._task_queue: Queue[Response] = Queue(maxsize=1)
        self._image_queue: Queue[ChallengeImage] = Queue()
        self.cr_queue: Queue[ChallengeResp] = Queue()

        self.cr: ChallengeResp = ChallengeResp()

        self._enable_evnet_listener(self.page)

    def _enable_evnet_listener(self, page: Page):
        page.on("response", self._task_handler)

    @logger.catch
    async def _task_handler(self, response: Response):
        if "/getcaptcha/" in response.url:
            # reset state
            while not self._image_queue.empty():
                self._image_queue.get_nowait()
            if self._task_queue.full():
                self._task_queue.get_nowait()

            # drop task
            self._task_queue.put_nowait(response)

        # /cr 在 Submit Event 之后，cr 截至目前是明文数据
        elif "/checkcaptcha/" in response.url:
            try:
                metadata = await response.json()
                self.cr_queue.put_nowait(ChallengeResp(**metadata))
            except Exception as err:
                logger.exception(err)

        # Image GET Event 发生在 /GetCaptcha 之后，此时假设 prompt 和 label 已被正确初始化
        # _image_handler 和 _task_cr_handler 以协程方式运行，但在业务逻辑上，_task_cr_handler 先发生。
        elif response.url.startswith("https://imgs3.hcaptcha.com/tip/"):
            image_bytes = await response.body()
            mime_type = await response.header_value("content-type")
            image_url = response.url

            suffix = ".jpeg"
            if isinstance(mime_type, str):
                _suffix = mime_type.split("/")[-1]
                if _suffix in ["jpg", "jpeg", "png", "webp"]:
                    suffix = f".{_suffix}"

            fn = f"{hashlib.md5(image_bytes).hexdigest()}{suffix}"
            fp = self._cache_dir / fn
            fp.write_bytes(image_bytes)

            # waiting for lock
            element = ChallengeImage(
                datapoint_uri=image_url, filename=fn, body=image_bytes, runtime_fp=fp
            )
            self._image_queue.put_nowait(element)

    @logger.catch
    async def _fetch_qr_data(self) -> OminousLand | None:
        qr_data = await self._task_queue.get()

        driver_conf = {
            "page": self.page,
            "tmp_dir": self.tmp_dir,
            "image_queue": self._image_queue,
            "ms": self.robotic_arm,
        }
        runnable: OminousLand | None = None

        match content_type := qr_data.headers.get("content-type"):
            case "application/octet-stream":
                driver_conf["inputs"] = await qr_data.body()
                runnable = ChallengeProblemDecoder.draws_from(**driver_conf)
            case "application/json":
                data = await qr_data.json()
                if data.get("pass"):
                    self.cr_queue.put_nowait(ChallengeResp(**data))
                else:
                    driver_conf["inputs"] = data
                    runnable = ChallengeProblemParser.draws_from(**driver_conf)
            case _:
                raise ValueError(f"Unknown Challenge Response Protocol - {content_type=}")

        return runnable

    @logger.catch
    async def _invoke_solver(self, runnable: OminousLand | None):
        if isinstance(runnable, OminousLand):
            await runnable.invoke(execution=self._tool_type)

    async def wait_for_challenge(
        self,
        execution_timeout: float = 90,
        response_timeout: float = 30.0,
        retry_on_failure: bool = True,
    ) -> ChallengeSignal:
        self._tool_type = ToolExecution.CHALLENGE

        # CoroutineTask: Assigning human-computer challenge tasks to the main thread coroutine.
        # Wait for the task to finish executing
        try:
            runnable = await self._fetch_qr_data()
            if not isinstance(runnable, OminousLand):
                logger.error("qr_data not found")
                return ChallengeSignal.QR_DATA_NOT_FOUND
            await asyncio.wait_for(self._invoke_solver(runnable), timeout=execution_timeout)
        except asyncio.TimeoutError:
            logger.error("Challenge execution timed out", timeout=execution_timeout)
            return ChallengeSignal.EXECUTION_TIMEOUT

        logger.debug("Invoke done", _tool_type=self._tool_type)

        # CoroutineTask: Assigned a new task
        # The possible reason is that the challenge was **manually** refreshed during the task.
        while self.cr_queue.empty():
            if not self._task_queue.empty():
                return await self.wait_for_challenge(execution_timeout, response_timeout)
            await asyncio.sleep(0.01)

        # CoroutineTask: Waiting for hCAPTCHA response processing result
        # After the completion of the human-machine challenge workflow,
        # it is expected to obtain a signal indicating whether the challenge was successful in the cr_queue.
        self.cr = ChallengeResp()
        try:
            self.cr = await self.cr_queue.get()
        except asyncio.TimeoutError:
            logger.error("Timeout waiting for challenge response", timeout=response_timeout)
            return ChallengeSignal.TIMEOUT
        else:
            # Match: Timeout / Loss
            if not self.cr or not self.cr.is_pass:
                if retry_on_failure:
                    logger.error("Invoke verification", is_pass=self.cr.is_pass)
                    return await self.wait_for_challenge(
                        execution_timeout, response_timeout, retry_on_failure=retry_on_failure
                    )
                return ChallengeSignal.RETRY
            if self.cr.is_pass:
                logger.success("Invoke verification", **self.cr.model_dump(by_alias=True))
                return ChallengeSignal.SUCCESS

    async def wait_for_collect(
        self, point: CollectibleType | None = None, *, batch: int = 20, timeout: float = 30.0
    ):
        self._tool_type = ToolExecution.COLLECT

        sitelink = Collectible(point=point).fixed_sitelink

        await self.page.goto(sitelink)
        await self.robotic_arm.click_checkbox()

        logger.debug("run collector", url=self.page.url)

        if batch >= 1:
            for i in range(1, batch + 1):
                with suppress(asyncio.TimeoutError):
                    await asyncio.wait_for(self._fetch_qr_data(), timeout=timeout)
                if not await self.robotic_arm.refresh_challenge():
                    return await self.wait_for_collect(point=point, batch=batch - i)

        logger.success("The dataset collection is complete.", sitelink=sitelink)
