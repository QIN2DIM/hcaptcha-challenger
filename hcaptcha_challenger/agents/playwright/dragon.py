# -*- coding: utf-8 -*-
# Time       : 2024/4/7 11:43
# Author     : QIN2DIM
# GitHub     : https://github.com/QIN2DIM
# Description:
import abc
import asyncio
import hashlib
import re
import shutil
from abc import ABC
from asyncio import Queue
from contextlib import suppress
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import List

from loguru import logger
from playwright.async_api import Page, Response, TimeoutError, expect

from hcaptcha_challenger.models import (
    ChallengeResp,
    QuestionResp,
    RequestType,
    Status,
    ChallengeImage,
    ToolExecution,
    CollectibleType,
    Collectible,
)
from hcaptcha_challenger.onnx.modelhub import ModelHub
from hcaptcha_challenger.tools.prompt_handler import handle

HOOK_PURCHASE = "//div[@id='webPurchaseContainer']//iframe"
HOOK_CHECKBOX = "//iframe[contains(@title, 'checkbox for hCaptcha')]"
HOOK_CHALLENGE = "//iframe[contains(@title, 'hCaptcha challenge')]"


@dataclass
class MechanicalSkeleton:
    page: Page

    async def click_checkbox(self):
        try:
            checkbox = self.page.frame_locator("//iframe[contains(@title,'checkbox')]")
            await checkbox.locator("#checkbox").click()
        except TimeoutError as err:
            logger.warning("Failed to click checkbox", reason=err)

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


@dataclass
class OminousLand(ABC):
    """不祥之地：base"""

    page: Page
    tmp_dir: Path
    ms: MechanicalSkeleton
    image_queue: Queue

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
        cls, page: Page, inputs: dict | bytes, tmp_dir: Path, image_queue: Queue, **kwargs
    ):
        # Cache images
        if not isinstance(tmp_dir, Path):
            tmp_dir = Path("tmp_dir")

        typed_dir = tmp_dir / "typed_dir"
        canvas_screenshot_dir = tmp_dir / "canvas_screenshot"

        monster = cls(
            page=page,
            ms=MechanicalSkeleton(page),
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

    def _init_imgdb(self, label: str, prompt: str):
        """run after _get_captcha"""
        self.tasklist.clear()
        self.examples.clear()

        inv = {"\\", "/", ":", "*", "?", "<", ">", "|", "\n"}
        for c in inv:
            label = label.replace(c, "")
            prompt = prompt.replace(c, "")
        label = label.strip()

        self.typed_dir = self.tmp_dir.joinpath(self.qr.request_type, label)
        self.typed_dir.mkdir(parents=True, exist_ok=True)

        if self.qr.request_type != RequestType.ImageLabelBinary:
            self.canvas_screenshot_dir = self.tmp_dir.joinpath(f"canvas_screenshot/{prompt}")
            self.canvas_screenshot_dir.mkdir(parents=True, exist_ok=True)

    async def _recall_tasklist(self):
        """run after _init_imgdb"""
        frame_challenge = self.ms.switch_to_challenge_frame()

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

            while not self.image_queue.empty():
                challenge_image: ChallengeImage = self.image_queue.get_nowait()
                challenge_images[challenge_image.datapoint_uri] = challenge_image

            for url in background_urls:
                challenge_image = challenge_images.get(url)
                if challenge_image:
                    self.tasklist.append(challenge_image)
                    if not self.typed_dir.joinpath(challenge_image.filename).exists():
                        shutil.move(src=challenge_image.runtime_fp, dst=self.typed_dir)

        elif self.qr.request_type == RequestType.ImageLabelAreaSelect:
            # For the object detection task, tasklist is only used to collect datasets.
            # The challenge in progress uses a canvas screenshot, not the challenge-image
            canvas_bgk = frame_challenge.locator("//div[class='bounding-box-example']")
            await expect(canvas_bgk).not_to_be_attached()

            # Expect only 1 image in the image_queue
            while not self.image_queue.empty():
                challenge_image: ChallengeImage = self.image_queue.get_nowait()
                if not self.typed_dir.joinpath(challenge_image.filename).exists():
                    shutil.move(src=challenge_image.runtime_fp, dst=self.typed_dir)
                # Cache image sequences for subsequent browser operations
                self.tasklist.append(challenge_image)

    @abc.abstractmethod
    async def _get_captcha(self, **kwargs):
        raise NotImplementedError

    async def _solve_captcha(self, **kwargs):
        frame_challenge = self.ms.switch_to_challenge_frame()

        match self.qr.request_type:
            case RequestType.ImageLabelBinary:
                pass
            case RequestType.ImageLabelAreaSelect:
                # Cache canvas to prepare for subsequent model processing
                # canvas = frame_challenge.locator("//canvas")
                # fp = self.canvas_screenshot_dir / f"{challenge_image.filename}.png"
                # await canvas.screenshot(type="png", path=fp, scale="css")
                pass
            case RequestType.ImageLabelMultipleChoice:
                pass
            case _:
                logger.warning("[INTERRUPT]", reason="Unknown type of challenge")

    async def _collect(self):
        await self._get_captcha()

        logger.debug(
            "task",
            label=self.label,
            type=self.qr.request_type,
            requester_question=self.qr.requester_question,
            trigger=self.__class__.__name__,
        )

        self._init_imgdb(self.label, self.prompt)
        await self._recall_tasklist()

    async def _challenge(self):
        await self._collect()
        await self._solve_captcha()

    async def invoke(self, execution: ToolExecution = ToolExecution.CHALLENGE):
        match execution:
            case ToolExecution.COLLECT:
                await self._collect()
            case ToolExecution.CHALLENGE:
                await self._challenge()


@dataclass
class ScarletWhisker(OminousLand):
    """赤髯 (Chì Rán) ->> json"""

    async def _get_captcha(self, **kwargs):
        self.qr = QuestionResp(**self.qr_data)

        self.prompt = self.qr.requester_question.get("en")
        self.label = handle(self.prompt)


@dataclass
class DemonLordOfHuiYue(OminousLand):
    """晦月魔君 ->> bytes"""

    async def _get_captcha(self, **kwargs):
        self.qr = QuestionResp()

        # IMPORTANT
        await self.page.wait_for_timeout(2000)

        frame_challenge = self.ms.switch_to_challenge_frame()

        # requester_question
        prompt_element = frame_challenge.locator("//h2[@class='prompt-text']")
        self.prompt = await prompt_element.text_content()
        self.label = handle(self.prompt)
        lang = await frame_challenge.locator("//html").get_attribute("lang")
        self.qr.requester_question[lang] = self.prompt

        # request_type
        if await frame_challenge.locator("//div[@class='task-grid']").count():
            self.qr.request_type = RequestType.ImageLabelBinary.value
            has_exp = await frame_challenge.locator("//div[@class='challenge-example']").count()
        elif await frame_challenge.locator("//div[contains(@class, 'bounding-box')]").count():
            self.qr.request_type = RequestType.ImageLabelAreaSelect.value
        else:
            # todo image_label_multiple_choice
            self.qr.request_type = RequestType.ImageLabelMultipleChoice.value


@dataclass
class AgentV:
    page: Page

    modelhub: ModelHub

    ms: MechanicalSkeleton = field(default_factory=MechanicalSkeleton)

    cr: ChallengeResp = field(default_factory=ChallengeResp)

    task_queue: Queue[Response] | None = None
    cr_queue: Queue[ChallengeResp] = field(default_factory=Queue)

    tmp_dir: Path = field(default_factory=Path)

    image_queue: Queue = field(default_factory=Queue)

    _tool_type: ToolExecution | None = None

    def __post_init__(self):
        # Control models
        self.modelhub = self.modelhub or ModelHub.from_github_repo()
        if not self.modelhub.label_alias:
            self.modelhub.parse_objects()

        self.tmp_dir = self.tmp_dir or Path("tmp_dir")
        self._cache_dir = self.tmp_dir / ".cache"
        self._cache_dir.mkdir(parents=True, exist_ok=True)

        self._enable_evnet_listener(self.page)

        self.task_queue = Queue(maxsize=1)

    @classmethod
    def into_solver(cls, page: Page, tmp_dir=None, modelhub: ModelHub | None = None, **kwargs):
        return cls(
            page=page, ms=MechanicalSkeleton(page), tmp_dir=tmp_dir, modelhub=modelhub, **kwargs
        )

    @property
    def status(self):
        return Status

    def _enable_evnet_listener(self, page: Page):
        page.on("response", self._task_handler)

    @logger.catch
    async def _task_handler(self, response: Response):
        if "/getcaptcha/" in response.url:
            # reset state
            while not self.image_queue.empty():
                self.image_queue.get_nowait()
            if self.task_queue.full():
                self.task_queue.get_nowait()

            # drop task
            self.task_queue.put_nowait(response)

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
            self.image_queue.put_nowait(element)

    @logger.catch
    async def _tool_execution(self):
        qr_data = await self.task_queue.get()

        driver_conf = {"page": self.page, "tmp_dir": self.tmp_dir, "image_queue": self.image_queue}
        runnable: OminousLand | None = None

        match content_type := qr_data.headers.get("content-type"):
            case "application/octet-stream":
                driver_conf["inputs"] = await qr_data.body()
                runnable = DemonLordOfHuiYue.draws_from(**driver_conf)
            case "application/json":
                data = await qr_data.json()
                if data.get("pass"):
                    self.cr_queue.put_nowait(ChallengeResp(**data))
                else:
                    driver_conf["inputs"] = data
                    runnable = ScarletWhisker.draws_from(**driver_conf)
            case _:
                raise ValueError(f"Unknown Challenge Response Protocol - {content_type=}")

        if isinstance(runnable, OminousLand):
            await runnable.invoke(execution=self._tool_type)

    async def wait_for_challenge(
        self, execution_timeout: float = 150.0, response_timeout: float = 30.0
    ) -> Status:
        self._tool_type = ToolExecution.CHALLENGE

        # CoroutineTask: Assigning human-computer challenge tasks to the main thread coroutine.
        # Wait for the task to finish executing
        try:
            await asyncio.wait_for(self._tool_execution(), timeout=execution_timeout)
        except asyncio.TimeoutError:
            logger.error("Challenge execution timed out", timeout=execution_timeout)
            return self.status.CHALLENGE_EXECUTION_TIMEOUT

        # CoroutineTask: Waiting for hCAPTCHA response processing result
        # After the completion of the human-machine challenge workflow,
        # it is expected to obtain a signal indicating whether the challenge was successful in the cr_queue.
        self.cr = ChallengeResp()
        try:
            self.cr = await self.cr_queue.get()
        except asyncio.TimeoutError:
            logger.error("Timeout waiting for challenge response", timeout=response_timeout)
            return self.status.CHALLENGE_RESPONSE_TIMEOUT
        else:
            logger.debug("[DONE]", **self.cr.model_dump(by_alias=True))

            # Match: Timeout / Loss
            if not self.cr or not self.cr.is_pass:
                return self.status.CHALLENGE_RETRY
            if self.cr.is_pass:
                return self.status.CHALLENGE_SUCCESS

    async def wait_for_collect(
        self, point: CollectibleType | None = None, *, batch: int = 20, timeout: float = 30.0
    ):
        self._tool_type = ToolExecution.COLLECT

        sitelink = Collectible(point=point).fixed_sitelink

        await self.page.goto(sitelink)
        await self.ms.click_checkbox()

        logger.debug("run collector", url=self.page.url)

        if batch >= 1:
            for i in range(1, batch + 1):
                with suppress(asyncio.TimeoutError):
                    await asyncio.wait_for(self._tool_execution(), timeout=timeout)
                if not await self.ms.refresh_challenge():
                    return await self.wait_for_collect(point=point, batch=batch - i)

        logger.success("The dataset collection is complete.", sitelink=sitelink)
