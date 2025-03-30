# -*- coding: utf-8 -*-
# Time       : 2024/4/7 11:43
# Author     : QIN2DIM
# GitHub     : https://github.com/QIN2DIM
# Description:
import asyncio
from asyncio import Queue
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path

from loguru import logger
from playwright.async_api import Page, Response, TimeoutError
from undetected_playwright.async_api import Locator

from hcaptcha_challenger.models import ChallengeResp, RequestType
from hcaptcha_challenger.tools import GeminiImageClassifier

HOOK_PURCHASE = "//div[@id='webPurchaseContainer']//iframe"
HOOK_CHALLENGE = "//iframe[contains(@title, 'hCaptcha challenge')]"


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


class TaskPayloadType(str, Enum):
    PLAINTEXT = "plaintext"
    CIPHERTEXT = "ciphertext"


@dataclass
class AgentConfig:
    GEMINI_API_KEY: str = ""
    cache_dir: Path = Path("tmp/.cache")

    execution_timeout: float = 90.0
    response_timeout: float = 30.0
    retry_on_failure: bool = True


class RoboticArm:

    def __init__(self, page: Page, config: AgentConfig):
        self.page = page
        self.config = config

        self._gic = GeminiImageClassifier(gemini_api_key=self.config.GEMINI_API_KEY)

    @property
    def checkbox_selector(self) -> str:
        return "//iframe[starts-with(@src,'https://newassets.hcaptcha.com/captcha/v1/') and contains(@src, 'frame=checkbox')]"

    @property
    def challenge_selector(self) -> str:
        return "//iframe[starts-with(@src,'https://newassets.hcaptcha.com/captcha/v1/') and contains(@src, 'frame=challenge')]"

    async def click_by_mouse(self, locator: Locator):
        bbox = await locator.bounding_box()

        center_x = bbox['x'] + bbox['width'] / 2
        center_y = bbox['y'] + bbox['height'] / 2

        await self.page.mouse.move(center_x, center_y)

        await self.page.mouse.click(center_x, center_y, delay=150)

    async def click_checkbox(self):
        checkbox_frame = self.page.frame_locator(self.checkbox_selector)
        checkbox_element = checkbox_frame.locator("//div[@id='checkbox']")
        await self.click_by_mouse(checkbox_element)

    async def refresh_challenge(self):
        try:
            refresh_frame = self.page.frame_locator(self.challenge_selector)
            refresh_element = refresh_frame.locator("//div[@class='refresh button']")
            await self.click_by_mouse(refresh_element)
        except TimeoutError as err:
            logger.warning(f"Failed to click refresh button - {err=}")

    def switch_to_challenge_frame(self, window: str = "login"):
        if window == "login":
            frame_challenge = self.page.frame_locator(self.challenge_selector)
        else:
            frame_purchase = self.page.frame_locator(HOOK_PURCHASE)
            frame_challenge = frame_purchase.frame_locator(self.challenge_selector)

        return frame_challenge

    async def check_crumb_count(self):
        """二分类任务中的翻页"""
        frame_challenge = self.switch_to_challenge_frame()
        crumbs = frame_challenge.locator("//div[@class='Crumb']")
        return 2 if await crumbs.first.is_visible() else 1

    async def check_challenge_type(self) -> RequestType:
        await self.page.wait_for_selector(self.challenge_selector)

        frame_challenge = self.switch_to_challenge_frame()
        samples = frame_challenge.locator("//div[@class='task-image']")
        count = await samples.count()
        if isinstance(count, int) and count == 9:
            return RequestType.ImageLabelBinary
        if isinstance(count, int) and count == 0:
            return RequestType.ImageLabelAreaSelect

        # todo: multiple

    async def challenge_image_label_binary(self):
        frame_challenge = self.switch_to_challenge_frame()
        crumb_count = await self.check_crumb_count()

        for _ in range(crumb_count):
            await self.page.wait_for_timeout(1000)

            # Get challenge-view
            challenge_view = frame_challenge.locator("//div[@class='challenge-view']")
            cache_dir = self.config.cache_dir.joinpath("challenge_view")
            current_time = datetime.now().strftime("%Y%m%d%H%M%S%f")
            cache_path = cache_dir.joinpath(f"{current_time}.png")
            await challenge_view.screenshot(type="png", path=cache_path)

            # Image classification
            results = self._gic.invoke(challenge_screenshot=cache_path)
            boolean_matrix = results.convert_box_to_boolean_matrix()

            logger.debug(f"Challenge Prompt: {results.challenge_prompt}")
            logger.debug(f"Coordinates: {results.coordinates}")
            logger.debug(f"Results: {boolean_matrix}")

            # drive the browser to work on the challenge
            positive_cases = 0
            for i, should_be_clicked in enumerate(boolean_matrix):
                if should_be_clicked:
                    xpath = f"//div[@class='task' and contains(@aria-label, '{i+1}')]"
                    task_image = frame_challenge.locator(xpath)
                    await self.click_by_mouse(task_image)
                    positive_cases += 1
                elif positive_cases == 0 and i == len(boolean_matrix) - 1:
                    xpath = f"//div[@class='task' and contains(@aria-label, '1')]"
                    task_image = frame_challenge.locator(xpath)
                    await self.click_by_mouse(task_image)

            # {{< Verify >}}
            with suppress(TimeoutError):
                submit_btn = frame_challenge.locator("//div[@class='button-submit button']")
                await self.click_by_mouse(submit_btn)


class AgentV:

    def __init__(self, page: Page, agent_config: AgentConfig):
        self.page = page
        self.config = agent_config

        self.robotic_arm = RoboticArm(page=page, config=agent_config)

        self._task_queue: Queue[Response] = Queue()
        self._challenge_resp_queue: Queue[ChallengeResp] = Queue()

        self.page.on("response", self._task_handler)

    @logger.catch
    async def _task_handler(self, response: Response):
        # /cr 在 Submit Event 之后，cr 截至目前是明文数据
        if "/getcaptcha/" in response.url:
            self._task_queue.put_nowait(response)
        elif "/checkcaptcha/" in response.url:
            try:
                metadata = await response.json()
                self._challenge_resp_queue.put_nowait(ChallengeResp(**metadata))
            except Exception as err:
                logger.exception(err)

    async def _check_pre_bypass(self) -> bool:
        the_latest_task = None
        while not self._task_queue.empty():
            the_latest_task = self._task_queue.get_nowait()

        if (
            the_latest_task
            and the_latest_task.headers.get("content-type", "") == "application/json"
        ):
            data = await the_latest_task.json()
            if data.get("pass"):
                self._challenge_resp_queue.put_nowait(ChallengeResp(**data))
                return True

        return False

    async def _solve_captcha(self):
        challenge_type = await self.robotic_arm.check_challenge_type()
        logger.debug(f"challenge_type: {challenge_type.value}")

        match challenge_type:
            case RequestType.ImageLabelBinary:
                try:
                    await self.robotic_arm.challenge_image_label_binary()
                except Exception as err:
                    logger.error(f"An error occurred while processing the challenge task - {err=}")
                    await self.robotic_arm.refresh_challenge()
            # todo NotSupported ImageLabelAreaSelect
            case RequestType.ImageLabelAreaSelect:
                await self.robotic_arm.refresh_challenge()
                await self.page.wait_for_timeout(2000)
                return await self._solve_captcha()  # fixme
            # todo NotSupported ImageLabelAreaSelect
            case RequestType.ImageLabelMultipleChoice:
                await self.robotic_arm.refresh_challenge()
                await self.page.wait_for_timeout(2000)
                return await self._solve_captcha()  # fixme
            case _:
                logger.error("[INTERRUPT]", reason="Unknown type of challenge")

    async def wait_for_challenge(self) -> ChallengeSignal:
        execution_timeout = self.config.execution_timeout
        response_timeout = self.config.response_timeout
        retry_on_failure = self.config.retry_on_failure

        # Assigning human-computer challenge tasks to the main thread coroutine.
        # ----------------------------------------------------------------------
        try:
            is_pre_bypass = await self._check_pre_bypass()
            if not is_pre_bypass:
                await asyncio.wait_for(self._solve_captcha(), timeout=execution_timeout)
        except asyncio.TimeoutError:
            logger.error("Challenge execution timed out", timeout=execution_timeout)
            return ChallengeSignal.EXECUTION_TIMEOUT

        logger.debug("Invoke done", _tool_type="challenge")

        # fixme debugger
        await self.page.pause()

        # Assigned a new task
        # -------------------
        # The possible reason is that the challenge was **manually** refreshed during the task.
        while self._challenge_resp_queue.empty():
            if not self._task_queue.empty():
                return await self.wait_for_challenge()
            await asyncio.sleep(0.01)

        # Waiting for hCAPTCHA response processing result
        # -----------------------------------------------
        # After the completion of the human-machine challenge workflow,
        # it is expected to obtain a signal indicating whether the challenge was successful in the cr_queue.
        self.cr = ChallengeResp()
        try:
            self.cr = await self._challenge_resp_queue.get()
        except asyncio.TimeoutError:
            logger.error("Timeout waiting for challenge response", timeout=response_timeout)
            return ChallengeSignal.TIMEOUT
        else:
            # Match: Timeout / Loss
            if not self.cr or not self.cr.is_pass:
                if retry_on_failure:
                    logger.error("Invoke verification", is_pass=self.cr.is_pass)
                    return await self.wait_for_challenge()
                return ChallengeSignal.RETRY
            if self.cr.is_pass:
                logger.success("Invoke verification", **self.cr.model_dump(by_alias=True))
                return ChallengeSignal.SUCCESS
