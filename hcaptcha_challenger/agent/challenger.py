# -*- coding: utf-8 -*-
# Time       : 2024/4/7 11:43
# Author     : QIN2DIM
# GitHub     : https://github.com/QIN2DIM
# Description:
import asyncio
import json
import os
import re
from asyncio import Queue
from contextlib import suppress
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import List, Any

from loguru import logger
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from undetected_playwright.async_api import Locator, expect, Page, Response, TimeoutError

from hcaptcha_challenger.models import CaptchaResponse, RequestType
from hcaptcha_challenger.tools import GeminiImageClassifier


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
    RETRY = "retry"
    QR_DATA_NOT_FOUND = "qr_data_not_found"
    EXECUTION_TIMEOUT = "challenge_execution_timeout"
    RESPONSE_TIMEOUT = "challenge_response_timeout"


class AgentConfig(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_ignore_empty=True, extra="ignore")

    GEMINI_API_KEY: str = Field(default_factory=lambda: os.environ.get("GEMINI_API_KEY", ""))
    cache_dir: Path = Path("tmp/.cache")
    captcha_response_dir: Path = Path("tmp/.captcha")

    execution_timeout: float = 90.0
    response_timeout: float = 30.0
    retry_on_failure: bool = True

    @field_validator('GEMINI_API_KEY', mode="after")
    @classmethod
    def validate_api_key(cls, v: Any) -> str:
        """
        Validates that the GEMINI_API_KEY is not empty.

        Args:
            v: The API key value to validate

        Returns:
            The validated API key

        Raises:
            ValueError: If the API key is empty
        """
        if not v or not isinstance(v, str):
            raise ValueError(
                "GEMINI_API_KEY is required but not provided. "
                "Please either pass it directly or set the GEMINI_API_KEY environment variable."
            )
        return v


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

    async def check_crumb_count(self):
        """Page turn in tasks"""
        frame_challenge = self.page.frame_locator(self.challenge_selector)
        crumbs = frame_challenge.locator("//div[@class='Crumb']")
        return 2 if await crumbs.first.is_visible() else 1

    async def check_challenge_type(self) -> RequestType:
        await self.page.wait_for_selector(self.challenge_selector)

        frame_challenge = self.page.frame_locator(self.challenge_selector)
        samples = frame_challenge.locator("//div[@class='task-image']")
        count = await samples.count()
        if isinstance(count, int) and count == 9:
            return RequestType.ImageLabelBinary
        if isinstance(count, int) and count == 0:
            return RequestType.ImageLabelAreaSelect

        # todo: multiple

    async def wait_for_all_loaders_complete(self):
        """Wait for all loading indicators to complete (become invisible)"""
        frame_challenge = self.page.frame_locator(self.challenge_selector)

        loading_indicators = frame_challenge.locator("//div[@class='loading-indicator']")
        count = await loading_indicators.count()

        if count == 0:
            logger.info("No load indicator found in the page")
            return True

        for i in range(count):
            loader = loading_indicators.nth(i)
            try:
                await expect(loader).to_have_attribute(
                    "style", re.compile(r"opacity:\s*0"), timeout=30000
                )
                await loading_indicators.nth(i).get_attribute("style")  # It cannot be removed
            except TimeoutError:
                logger.warning(f"The load indicator {i + 1}/{count} waits for a timeout")

        return True

    async def challenge_image_label_binary(self):
        frame_challenge = self.page.frame_locator(self.challenge_selector)
        crumb_count = await self.check_crumb_count()

        for _ in range(crumb_count):
            await self.wait_for_all_loaders_complete()

            # Get challenge-view
            challenge_view = frame_challenge.locator("//div[@class='challenge-view']")
            cache_dir = self.config.cache_dir.joinpath("challenge_view")
            current_time = datetime.now().strftime("%Y%m%d%H%M%S%f")
            cache_path = cache_dir.joinpath(f"{current_time}.png")
            await challenge_view.screenshot(type="png", path=cache_path)

            # Image classification
            results = self._gic.invoke(challenge_screenshot=cache_path)
            boolean_matrix = results.convert_box_to_boolean_matrix()

            logger.debug(f'ToolInvokeMessage: {results.log_message}')

            # drive the browser to work on the challenge
            positive_cases = 0
            xpath_task_image = "//div[@class='task' and contains(@aria-label, '{index}')]"
            for i, should_be_clicked in enumerate(boolean_matrix):
                if should_be_clicked:
                    task_image = frame_challenge.locator(xpath_task_image.format(index=i + 1))
                    await self.click_by_mouse(task_image)
                    positive_cases += 1
                elif positive_cases == 0 and i == len(boolean_matrix) - 1:
                    task_image = frame_challenge.locator(xpath_task_image.format(index=1))
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
        self._captcha_response_queue: Queue[CaptchaResponse] = Queue()
        self.cr_list: List[CaptchaResponse] = []

        self.page.on("response", self._task_handler)

    @logger.catch
    async def _task_handler(self, response: Response):
        if "/getcaptcha/" in response.url:
            self._task_queue.put_nowait(response)
        elif "/checkcaptcha/" in response.url:
            try:
                metadata = await response.json()
                self._captcha_response_queue.put_nowait(CaptchaResponse(**metadata))
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
                cr = CaptchaResponse(**data)
                self._captcha_response_queue.put_nowait(cr)
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
                    logger.error(f"ChallengeException - {err=}")
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

    def _cache_validated_captcha_response(self, cr: CaptchaResponse):
        if not cr.is_pass:
            return

        self.cr_list.append(cr)

        try:
            captcha_response = cr.model_dump(mode="json", by_alias=True)
            current_time = datetime.now().strftime("%Y%m%d/%Y%m%d%H%M%S%f")
            cache_path = self.config.captcha_response_dir.joinpath(f"{current_time}.json")
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            t = json.dumps(captcha_response, indent=2, ensure_ascii=False)
            cache_path.write_text(t, encoding="utf-8")
        except Exception as err:
            logger.error(f"Saving captcha response failed - {err}")

    async def wait_for_challenge(self) -> ChallengeSignal:
        # Assigning human-computer challenge tasks to the main thread coroutine.
        # ----------------------------------------------------------------------
        try:
            is_pre_bypass = await self._check_pre_bypass()
            if not is_pre_bypass:
                await asyncio.wait_for(self._solve_captcha(), timeout=self.config.execution_timeout)
        except asyncio.TimeoutError:
            logger.error("Challenge execution timed out", timeout=self.config.execution_timeout)
            return ChallengeSignal.EXECUTION_TIMEOUT

        # Waiting for hCAPTCHA response processing result
        # -----------------------------------------------
        # After the completion of the human-machine challenge workflow,
        # it is expected to obtain a signal indicating whether the challenge was successful in the cr_queue.
        logger.debug("Start checking captcha response")
        try:
            cr = await self._captcha_response_queue.get()
        except asyncio.TimeoutError:
            logger.error(f"Wait for captcha response timeout {self.config.response_timeout}s")
            return ChallengeSignal.EXECUTION_TIMEOUT
        else:
            # Match: Timeout / Loss
            if not cr or not cr.is_pass:
                if self.config.retry_on_failure:
                    logger.warning("Failed to challenge, try to retry the strategy")
                    await self.page.wait_for_timeout(1500)
                    _signal = await self._task_queue.get()
                    self._task_queue.put_nowait(_signal)
                    return await self.wait_for_challenge()
                return ChallengeSignal.FAILURE
            # Match: Success
            if cr.is_pass:
                logger.success("Challenge success")
                self._cache_validated_captcha_response(cr)
                return ChallengeSignal.SUCCESS
