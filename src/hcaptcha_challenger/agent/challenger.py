# -*- coding: utf-8 -*-
# Time       : 2024/4/7 11:43
# Author     : QIN2DIM
# GitHub     : https://github.com/QIN2DIM
# Description:
import asyncio
import json
import math
import os
import random
import re
from asyncio import Queue
from contextlib import suppress
from datetime import datetime
from pathlib import Path
from typing import Any
from typing import List, Tuple
from uuid import uuid4

import matplotlib.pyplot as plt
import msgpack
from loguru import logger
from playwright.async_api import Locator, expect, Page, Response, TimeoutError, FrameLocator, Frame
from pydantic import Field, field_validator, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict
from tenacity import retry, stop_after_attempt, wait_fixed

from hcaptcha_challenger.helper import create_coordinate_grid
from hcaptcha_challenger.models import (
    CaptchaResponse,
    RequestType,
    ChallengeSignal,
    SCoTModelType,
    DEFAULT_SCOT_MODEL,
    DEFAULT_FAST_SHOT_MODEL,
    FastShotModelType,
    SpatialPath,
    CaptchaPayload,
    IGNORE_REQUEST_TYPE_LITERAL,
    INV,
)
from hcaptcha_challenger.models import ChallengeTypeEnum, CoordinateGrid
from hcaptcha_challenger.skills import SkillManager
from hcaptcha_challenger.tools import (
    ImageClassifier,
    ChallengeRouter,
    SpatialPathReasoner,
    SpatialPointReasoner,
)


def _generate_bezier_trajectory(
    start: Tuple[float, float], end: Tuple[float, float], steps: int
) -> List[Tuple[float, float]]:
    """
    Generates a quadratic bezier curve trajectory between start and end points.
    """
    points = []

    # Calculate distance between points
    distance = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)

    # Create control point(s) for the bezier curve
    # For longer distances, we use a higher control point offset
    offset_factor = min(0.3, max(0.1, distance / 1000))

    # Random control point that's offset from the midpoint
    mid_x = (start[0] + end[0]) / 2
    mid_y = (start[1] + end[1]) / 2

    # Create slight randomness in the control point
    control_x = mid_x + random.uniform(-1, 1) * distance * offset_factor
    control_y = mid_y + random.uniform(-1, 1) * distance * offset_factor

    # Generate points along the bezier curve
    for i in range(steps + 1):
        t = i / steps
        # Quadratic bezier formula
        x = (1 - t) ** 2 * start[0] + 2 * (1 - t) * t * control_x + t**2 * end[0]
        y = (1 - t) ** 2 * start[1] + 2 * (1 - t) * t * control_y + t**2 * end[1]
        points.append((x, y))

    return points


def _generate_dynamic_delays(steps: int, base_delay: int) -> List[float]:
    """
    Generates dynamic delays between mouse movements to simulate human-like acceleration/deceleration.
    """
    delays = []

    # Acceleration profile: slower at start and end, faster in the middle
    for i in range(steps + 1):
        progress = i / steps

        # Ease in-out function (slow start, fast middle, slow end)
        if progress < 0.5:
            factor = 2 * progress * progress  # Accelerate
        else:
            progress = progress - 1
            factor = 1 - (-2 * progress * progress)  # Decelerate

        # Adjust delay based on position in the curve (1.5x at ends, 0.6x in middle)
        delay_factor = 1.5 - 0.9 * factor

        # Add slight randomness to delays (±10%)
        random_factor = random.uniform(0.9, 1.1)

        delays.append(base_delay * delay_factor * random_factor)

    return delays


SINGLE_IGNORE_TYPE = IGNORE_REQUEST_TYPE_LITERAL | RequestType | ChallengeTypeEnum
IGNORE_REQUEST_TYPE_LIST = List[SINGLE_IGNORE_TYPE]


class AgentConfig(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_ignore_empty=True, extra="ignore")

    GEMINI_API_KEY: SecretStr = Field(
        default_factory=lambda: SecretStr(os.environ.get("GEMINI_API_KEY", "")),
        description="Create API Key https://aistudio.google.com/app/apikey",
    )

    cache_dir: Path = Path("tmp/.cache")
    challenge_dir: Path = Path("tmp/.challenge")
    captcha_response_dir: Path = Path("tmp/.captcha")
    ignore_request_types: IGNORE_REQUEST_TYPE_LIST | None = Field(default_factory=list)
    ignore_request_questions: List[str] | None = Field(default_factory=list)

    DISABLE_BEZIER_TRAJECTORY: bool = Field(
        default=False,
        description="If you use Camoufox, it is recommended to turn off "
        "the custom Bessel track generator of hcaptcha-challenger "
        "and use Camoufox(humanize=True)",
    )

    MAX_CRUMB_COUNT: int = Field(
        default=2,
        description="""
        CRUMB_COUNT: The number of challenge rounds you need to solve once the challenge starts.
        In the vast majority of cases this value will be 2, some specialized sites will set this value to 3.
        In most cases you don't need to change this value, the `_review_challenge_type` task determines the exact value of `CRUMB_COUNT` based on the information of the assigned task.
        Only manually change this value if you are working on a very specific task that prevents the `_review_challenge_type` from hijacking the task information and the maximum number of tasks > 2.
        """,
    )

    EXECUTION_TIMEOUT: float = Field(
        default=120,
        description="When your local network is poor, increase this value appropriately [unit: second]",
    )
    RESPONSE_TIMEOUT: float = Field(
        default=30,
        description="When your local network is poor, increase this value appropriately [unit: second]",
    )
    RETRY_ON_FAILURE: bool = Field(
        default=True, description="Re-execute the challenge when it fails"
    )
    WAIT_FOR_CHALLENGE_VIEW_TO_RENDER_MS: int = Field(
        default=1500,
        description="When your local network is poor, increase this value appropriately [unit: millisecond]",
    )

    CHALLENGE_CLASSIFIER_MODEL: FastShotModelType = Field(
        default=DEFAULT_FAST_SHOT_MODEL,
        description="For the challenge classification task \n"
        "Used as last resort when HSW decoding fails.",
    )
    IMAGE_CLASSIFIER_MODEL: SCoTModelType = Field(
        default=DEFAULT_SCOT_MODEL, description="For the challenge type: `image_label_binary`"
    )
    SPATIAL_POINT_REASONER_MODEL: SCoTModelType = Field(
        default=DEFAULT_SCOT_MODEL,
        description="For the challenge type: `image_label_area_select` (single/multi)",
    )
    SPATIAL_PATH_REASONER_MODEL: SCoTModelType = Field(
        default=DEFAULT_SCOT_MODEL,
        description="For the challenge type: `image_drag_drop` (single/multi)",
    )

    coordinate_grid: CoordinateGrid | None = Field(default_factory=CoordinateGrid)

    enable_challenger_debug: bool | None = Field(default=False, description="Enable debug mode")

    # == Skills Configuration == #
    custom_skills_path: Path | None = Field(
        default=None, description="Path to custom skills rules.yaml"
    )
    enable_skills_update: bool = Field(
        default=True, description="Enable auto-update of skills from GitHub"
    )
    skills_update_repo: str = Field(
        default="QIN2DIM/hcaptcha-challenger", description="GitHub repo for skills update"
    )
    skills_update_branch: str = Field(default="main", description="GitHub branch for skills update")

    @field_validator('GEMINI_API_KEY', mode="before")
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
                "Create API Key -> https://aistudio.google.com/app/apikey"
            )
        return v

    @property
    def spatial_grid_cache(self):
        return self.cache_dir.joinpath("spatial_grid")

    def create_cache_key(
        self,
        captcha_payload: CaptchaPayload | None = None,
        request_type: str = "type",
        prompt: str = "unknown",
    ) -> Path:
        """

        Args:
            captcha_payload:
            request_type:
            prompt:

        Returns: ./.challenge / require_type / prompt / current_time

        """
        current_datetime = datetime.now()
        current_time = current_datetime.strftime("%Y%m%d/%Y%m%d%H%M%S%f")

        prompt = prompt.translate(str.maketrans("", "", "".join(INV)))

        if not captcha_payload:
            _cache_key_temp = self.challenge_dir.joinpath(request_type, prompt, current_time)
            if self.enable_challenger_debug:
                logger.debug(f"Create cache-key [NotCaptchaPayload] - {_cache_key_temp.resolve()}")
            return _cache_key_temp

        cache_key = self.challenge_dir.joinpath(
            captcha_payload.request_type.value,
            captcha_payload.get_requester_question(),
            current_time,
        )

        try:
            _cache_path_captcha = cache_key.joinpath(f"{cache_key.name}_captcha.json")
            _cache_path_captcha.parent.mkdir(parents=True, exist_ok=True)

            _unpacked_data = captcha_payload.model_dump(mode="json")
            _cache_path_captcha.write_text(
                json.dumps(_unpacked_data, indent=2, ensure_ascii=False), encoding="utf8"
            )
        except Exception as e:
            logger.error(f"Failed to write captcha payload to cache: {e}")

        if self.enable_challenger_debug:
            logger.debug(f"Create cache-key [Direct] - {cache_key.resolve()}")

        return cache_key


class RoboticArm:

    def __init__(self, page: Page, config: AgentConfig):
        self.page = page
        self.config = config
        self._debug = config.enable_challenger_debug

        self._challenge_router = ChallengeRouter(
            gemini_api_key=self.config.GEMINI_API_KEY.get_secret_value(),
            model=self.config.CHALLENGE_CLASSIFIER_MODEL,
        )
        self._image_classifier = ImageClassifier(
            gemini_api_key=self.config.GEMINI_API_KEY.get_secret_value(),
            model=self.config.IMAGE_CLASSIFIER_MODEL,
        )
        self._spatial_path_reasoner = SpatialPathReasoner(
            gemini_api_key=self.config.GEMINI_API_KEY.get_secret_value(),
            model=self.config.SPATIAL_PATH_REASONER_MODEL,
        )
        self._spatial_point_reasoner = SpatialPointReasoner(
            gemini_api_key=self.config.GEMINI_API_KEY.get_secret_value(),
            model=self.config.SPATIAL_POINT_REASONER_MODEL,
        )
        self._skill_manager = SkillManager(agent_config=config)
        self.signal_crumb_count: int | None = None
        self.captcha_payload: CaptchaPayload | None = None
        self._challenge_prompt: str | None = None

        self._checkbox_selector = "//iframe[starts-with(@src,'https://newassets.hcaptcha.com/captcha/v1/') and contains(@src, 'frame=checkbox')]"
        self._challenge_selector = "//iframe[starts-with(@src,'https://newassets.hcaptcha.com/captcha/v1/') and contains(@src, 'frame=challenge')]"

    @property
    def checkbox_selector(self) -> str:
        return self._checkbox_selector

    @property
    def challenge_selector(self) -> str:
        return self._challenge_selector

    async def get_challenge_frame_locator(self) -> Frame | None:
        candidate_frame = self._find_challenge_frame_recursive(self.page.main_frame, max_depth=4)

        if candidate_frame:
            with suppress(Exception):
                challenge_view = candidate_frame.locator("//div[@class='challenge-view']")
                is_visible = await challenge_view.is_visible(timeout=1000)

                if is_visible:
                    return candidate_frame

        try:
            challenge_frames = []
            all_frames = self.page.frames
            for frame in all_frames:
                if (
                    frame.url.startswith("https://newassets.hcaptcha.com/captcha/v1/")
                    and "frame=challenge" in frame.url
                ):
                    challenge_frames.append(frame)

            for frame in challenge_frames:
                with suppress(Exception):
                    challenge_view = frame.locator("//div[@class='challenge-view']")
                    if await challenge_view.is_visible():
                        return frame
        except Exception as e:
            logger.error(f"Error finding all iframes: {e}")

        logger.error("Cannot find a valid challenge frame")
        return None

    def _find_challenge_frame_recursive(
        self, frame: Frame, current_depth=0, max_depth=4
    ) -> Frame | None:
        if current_depth >= max_depth:
            return None

        candidate_frames = []

        for child_frame in frame.child_frames:
            if (
                not child_frame.child_frames
                and child_frame.url.startswith("https://newassets.hcaptcha.com/captcha/v1/")
                and "frame=challenge" in child_frame.url
            ):
                candidate_frames.append(child_frame)
            else:
                found_in_child = self._find_challenge_frame_recursive(
                    child_frame, current_depth + 1, max_depth
                )
                if found_in_child:
                    return found_in_child

        if candidate_frames:
            return candidate_frames[0]

        return None

    def _match_user_prompt(self, job_type: ChallengeTypeEnum) -> str:
        try:
            challenge_prompt = (
                self.captcha_payload.get_requester_question()
                if self.captcha_payload
                else self._challenge_prompt
            )
            if challenge_prompt and isinstance(challenge_prompt, str):
                return self._skill_manager.get_skill(challenge_prompt, job_type)
        except Exception as e:
            logger.warning(f"Error while processing captcha payload: {e}")

        return f"Please note that the current task type is: {job_type.value}"

    async def click_by_mouse(self, locator: Locator):
        bbox = await locator.bounding_box()
        if bbox is None:
            raise ValueError("Element is not visible or does not exist")

        x: float = bbox['x']
        y: float = bbox['y']
        width: float = bbox['width']
        height: float = bbox['height']

        center_x = x + width / 2
        center_y = y + height / 2

        await self.page.mouse.move(center_x, center_y)

        await self.page.mouse.click(center_x, center_y, delay=150)

    async def click_checkbox(self):
        checkbox_frame = self.page.frame_locator(self.checkbox_selector)
        checkbox_element = checkbox_frame.locator("//div[@id='checkbox']")
        await self.click_by_mouse(checkbox_element)

    async def refresh_challenge(self):
        try:
            refresh_frame = await self.get_challenge_frame_locator()
            refresh_element = refresh_frame.locator("//div[@class='refresh button']")
            await self.click_by_mouse(refresh_element)
        except TimeoutError as err:
            logger.warning(f"Failed to click refresh button - {err=}")

    async def check_crumb_count(self):
        """Page turn in tasks"""
        # Determine the number of tasks based on hsw
        if isinstance(self.signal_crumb_count, int) and self.signal_crumb_count >= 1:
            return self.signal_crumb_count

        # Determine the number of tasks based on DOM
        await self.page.wait_for_timeout(500)
        frame_challenge = await self.get_challenge_frame_locator()
        crumbs = frame_challenge.locator("//div[@class='Crumb']")
        with suppress(Exception):
            crumbs_count = await crumbs.count()
            return crumbs_count if crumbs_count else 1
        return self.config.MAX_CRUMB_COUNT if await crumbs.first.is_visible() else 1

    async def check_challenge_type(self) -> RequestType | ChallengeTypeEnum | None:
        # fixme
        with suppress(Exception):
            await self.page.wait_for_selector(self.challenge_selector, timeout=1000)

        frame_challenge = await self.get_challenge_frame_locator()

        samples = frame_challenge.locator("//div[@class='task-image']")
        count = await samples.count()
        if isinstance(count, int) and count == 9:
            return RequestType.IMAGE_LABEL_BINARY
        if isinstance(count, int) and count == 0:
            tms = self.config.WAIT_FOR_CHALLENGE_VIEW_TO_RENDER_MS * 1.5
            await self.page.wait_for_timeout(tms)
            challenge_view = frame_challenge.locator("//div[@class='challenge-view']")
            cache_path = self.config.cache_dir.joinpath(f"challenge_view/_artifacts/{uuid4()}.png")
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            await challenge_view.screenshot(type="png", path=cache_path)
            router_result = await self._challenge_router(challenge_screenshot=cache_path)
            self._challenge_prompt = router_result.challenge_prompt
            return router_result.challenge_type
        return None

    async def _wait_for_all_loaders_complete(self):
        """Wait for all loading indicators to complete (become invisible)"""
        frame_challenge = await self.get_challenge_frame_locator()

        await self.page.wait_for_timeout(self.config.WAIT_FOR_CHALLENGE_VIEW_TO_RENDER_MS)

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
            except ValueError:
                # todo requires smarter waiting methods
                await self.page.wait_for_timeout(130)

        return True

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_fixed(1),
        before_sleep=lambda retry_state: logger.warning(
            f"Retry request ({retry_state.attempt_number}/2) - Wait 1 second - Exception: {retry_state.outcome.exception()}"
        ),
    )
    async def _capture_spatial_mapping(
        self, frame_challenge: FrameLocator | Frame, cache_key: Path, crumb_id: int | str
    ):
        # Capture challenge-view
        challenge_view = frame_challenge.locator("//div[@class='challenge-view']")
        challenge_screenshot = cache_key.joinpath(f"{cache_key.name}_{crumb_id}_challenge_view.png")
        challenge_screenshot.parent.mkdir(parents=True, exist_ok=True)
        await challenge_view.screenshot(type="png", path=challenge_screenshot)

        challenge_view = frame_challenge.locator("//div[@class='challenge-view']")
        bbox = await challenge_view.bounding_box()

        # Save grid field
        result = create_coordinate_grid(
            challenge_screenshot,
            bbox,
            x_line_space_num=self.config.coordinate_grid.x_line_space_num,
            y_line_space_num=self.config.coordinate_grid.y_line_space_num,
            color=self.config.coordinate_grid.color,
            adaptive_contrast=self.config.coordinate_grid.adaptive_contrast,
        )

        grid_divisions = cache_key.joinpath(f"{cache_key.name}_{crumb_id}_spatial_helper.png")
        grid_divisions.parent.mkdir(parents=True, exist_ok=True)
        plt.imsave(str(grid_divisions.resolve()), result)

        return challenge_screenshot, grid_divisions

    async def _perform_drag_drop(self, path: SpatialPath, steps: int = 25, delay_ms: int = 15):
        """
        Performs a human-like drag and drop operation using bezier curve trajectory.

        Args:
            path: The SpatialPath containing start and end coordinates
            steps: Number of intermediate steps for the mouse movement
            delay_ms: Base delay between steps in milliseconds
        """
        start_x, start_y = path.start_point.x, path.start_point.y
        end_x, end_y = path.end_point.x, path.end_point.y

        if self.config.DISABLE_BEZIER_TRAJECTORY:
            await self.page.mouse.move(start_x, start_y)
            await self.page.mouse.down()
            await self.page.mouse.move(end_x, end_y)
            await self.page.mouse.up()
            return

        # Move to the starting position
        await self.page.mouse.move(start_x, start_y)

        # Small random delay before pressing down (human reaction time)
        await asyncio.sleep(random.uniform(0.05, 0.15))

        # Press the mouse button down
        await self.page.mouse.down()

        # Generate a bezier curve path with a control point
        points = _generate_bezier_trajectory((start_x, start_y), (end_x, end_y), steps)

        # Add velocity variation (slow start, fast middle, slow end)
        delays = _generate_dynamic_delays(steps, base_delay=delay_ms)

        # Perform the drag with human-like movement
        for i, ((current_x, current_y), delay) in enumerate(zip(points, delays)):
            # Add slight "noise" to the path (more pronounced near the end)
            if i > steps * 0.7:  # In the last 30% of the movement
                # More micro-adjustments near the end
                noise_factor = 0.5 if i > steps * 0.9 else 0.2
                current_x += random.uniform(-noise_factor, noise_factor)
                current_y += random.uniform(-noise_factor, noise_factor)

            await self.page.mouse.move(current_x, current_y)
            await asyncio.sleep(delay / 1000)

        # Ensure we end exactly at the target position
        await self.page.mouse.move(end_x, end_y)

        # Small pause before releasing (human precision adjustment)
        await asyncio.sleep(random.uniform(0.05, 0.1))

        # Release the mouse button at the destination
        await self.page.mouse.up()

        # Small pause between drag operations
        await asyncio.sleep(random.uniform(0.08, 0.12))

    async def challenge_image_label_binary(self):
        frame_challenge = await self.get_challenge_frame_locator()
        crumb_count = await self.check_crumb_count()
        cache_key = self.config.create_cache_key(self.captcha_payload)

        for cid in range(crumb_count):
            await self._wait_for_all_loaders_complete()

            # Get challenge-view
            challenge_view = frame_challenge.locator("//div[@class='challenge-view']")
            challenge_screenshot = cache_key.joinpath(f"{cache_key.name}_{cid}_challenge_view.png")
            await challenge_view.screenshot(type="png", path=challenge_screenshot)

            # Image classification
            response = await self._image_classifier(challenge_screenshot=challenge_screenshot)
            boolean_matrix = response.convert_box_to_boolean_matrix()

            logger.debug(f'[{cid+1}/{crumb_count}]ToolInvokeMessage: {response.log_message}')
            self._image_classifier.cache_response(
                path=cache_key.joinpath(f"{cache_key.name}_{cid}_model_answer.json")
            )

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

    async def challenge_image_drag_drop(self, job_type: ChallengeTypeEnum):
        frame_challenge = await self.get_challenge_frame_locator()
        crumb_count = await self.check_crumb_count()
        cache_key = self.config.create_cache_key(self.captcha_payload)

        for cid in range(crumb_count):
            await self.page.wait_for_timeout(self.config.WAIT_FOR_CHALLENGE_VIEW_TO_RENDER_MS)

            raw, projection = await self._capture_spatial_mapping(frame_challenge, cache_key, cid)

            user_prompt = self._match_user_prompt(job_type)

            response = await self._spatial_path_reasoner(
                challenge_screenshot=raw,
                grid_divisions=projection,
                auxiliary_information=user_prompt,
            )
            logger.debug(f'[{cid+1}/{crumb_count}]ToolInvokeMessage: {response.log_message}')
            self._spatial_path_reasoner.cache_response(
                path=cache_key.joinpath(f"{cache_key.name}_{cid}_model_answer.json")
            )

            for path in response.paths:
                await self._perform_drag_drop(path)

            # {{< Verify >}}
            with suppress(TimeoutError):
                submit_btn = frame_challenge.locator("//div[@class='button-submit button']")
                await self.click_by_mouse(submit_btn)

    async def challenge_image_label_select(self, job_type: ChallengeTypeEnum):
        frame_challenge = await self.get_challenge_frame_locator()
        crumb_count = await self.check_crumb_count()
        cache_key = self.config.create_cache_key(self.captcha_payload)

        for cid in range(crumb_count):
            await self.page.wait_for_timeout(self.config.WAIT_FOR_CHALLENGE_VIEW_TO_RENDER_MS)

            raw, projection = await self._capture_spatial_mapping(frame_challenge, cache_key, cid)

            user_prompt = self._match_user_prompt(job_type)

            response = await self._spatial_point_reasoner(
                challenge_screenshot=raw,
                grid_divisions=projection,
                auxiliary_information=user_prompt,
            )
            logger.debug(f'[{cid+1}/{crumb_count}]ToolInvokeMessage: {response.log_message}')
            self._spatial_point_reasoner.cache_response(
                path=cache_key.joinpath(f"{cache_key.name}_{cid}_model_answer.json")
            )

            for point in response.points:
                await self.page.mouse.click(point.x, point.y, delay=180)
                await self.page.wait_for_timeout(500)

            # {{< Verify >}}
            with suppress(TimeoutError):
                submit_btn = frame_challenge.locator("//div[@class='button-submit button']")
                await self.click_by_mouse(submit_btn)


class AgentV:

    def __init__(self, page: Page, agent_config: AgentConfig):
        self.page = page
        self.config = agent_config

        self.robotic_arm = RoboticArm(page=page, config=agent_config)

        self._captcha_payload: CaptchaPayload | None = None
        self._captcha_payload_queue: Queue[CaptchaPayload | None] = Queue()
        self._captcha_response_queue: Queue[CaptchaResponse] = Queue()
        self.cr_list: List[CaptchaResponse] = []

        self.page.on("response", self._task_handler)

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

    @logger.catch
    async def _task_handler(self, response: Response):
        if response.url.endswith("/hsw.js"):
            try:
                hsw_text = await response.text()
                await self.page.evaluate(hsw_text)
                await self.page.evaluate(
                    """
                    () => {
                        return typeof hsw === 'function' ? true : 'hsw不是函数';
                    }
                    """
                )
            except Exception as err:
                logger.error(f"An error occurred while injecting hsw script: {err}")
        elif "/getcaptcha/" in response.url:
            self._captcha_payload = None

            # Content-Type: application/json
            if response.headers.get("content-type", "") == "application/json":
                data = await response.json()
                if data.get("pass"):
                    while not self._captcha_response_queue.empty():
                        self._captcha_response_queue.get_nowait()
                    cr = CaptchaResponse(**data)
                    self._captcha_response_queue.put_nowait(cr)
                    return
                if data.get("request_config"):
                    captcha_payload = CaptchaPayload(**data)
                    self._captcha_payload_queue.put_nowait(captcha_payload)
                    return

            # Content-Type: stream
            try:
                raw_data = await response.body()
                has_hsw = await self.page.evaluate(
                    """
                    () => {
                        return typeof hsw === 'function' ? true : false;
                    }
                    """
                )

                if has_hsw:
                    result = await self.page.evaluate(
                        f"""
                        async () => {{
                            const byteArray = new Uint8Array({list(raw_data)});
                            console.log('Data has been converted to Uint8Array, length:', byteArray.length);

                            try {{
                                const hswResult = await hsw(0, byteArray);
                                return Array.from(hswResult);
                            }} catch (e) {{
                                return {{error: e.toString()}};
                            }}
                        }}
                        """
                    )

                    if isinstance(result, list) and not any(
                        isinstance(x, dict) and "error" in x for x in result
                    ):
                        unpacked_data = msgpack.unpackb(bytes(result))
                        captcha_payload = CaptchaPayload(**unpacked_data)
                        self._captcha_payload_queue.put_nowait(captcha_payload)

                        return
                # If the reverse fails, fall back to the original process
                else:
                    logger.warning("HSW reverse failed, fallback to regular processing")
                    self._captcha_payload_queue.put_nowait(None)
            except Exception as err:
                logger.error(f"Reverse processing getcaptcha failed: {err}")
                self._captcha_payload_queue.put_nowait(None)
        elif "/checkcaptcha/" in response.url:
            try:
                metadata = await response.json()
                self._captcha_response_queue.put_nowait(CaptchaResponse(**metadata))
            except Exception as err:
                logger.exception(err)

    async def _review_challenge_type(self) -> RequestType | ChallengeTypeEnum:
        try:
            self._captcha_payload = await asyncio.wait_for(
                self._captcha_payload_queue.get(), timeout=30.0
            )
            await self.page.wait_for_timeout(500)
        except asyncio.TimeoutError:
            logger.error("Wait for captcha payload to timeout")
            self._captcha_payload = None

        self.robotic_arm.signal_crumb_count = None
        self.robotic_arm.captcha_payload = None
        if not self._captcha_payload:
            return await self.robotic_arm.check_challenge_type()

        try:
            request_type = self._captcha_payload.request_type
            tasklist = self._captcha_payload.tasklist
            tasklist_length = len(tasklist)
            self.robotic_arm.captcha_payload = self._captcha_payload
            match request_type:
                case RequestType.IMAGE_LABEL_BINARY:
                    self.robotic_arm.signal_crumb_count = int(tasklist_length / 9)
                    return RequestType.IMAGE_LABEL_BINARY
                case RequestType.IMAGE_LABEL_AREA_SELECT:
                    self.robotic_arm.signal_crumb_count = tasklist_length
                    max_shapes = self._captcha_payload.request_config.max_shapes_per_image
                    if not isinstance(max_shapes, int):
                        return await self.robotic_arm.check_challenge_type()
                    return (
                        ChallengeTypeEnum.IMAGE_LABEL_SINGLE_SELECT
                        if max_shapes == 1
                        else ChallengeTypeEnum.IMAGE_LABEL_MULTI_SELECT
                    )
                case RequestType.IMAGE_DRAG_DROP:
                    self.robotic_arm.signal_crumb_count = tasklist_length
                    return (
                        ChallengeTypeEnum.IMAGE_DRAG_SINGLE
                        if len(tasklist[0].entities) == 1
                        else ChallengeTypeEnum.IMAGE_DRAG_MULTI
                    )

            logger.warning(f"Unknown request_type: {request_type=}")
        except Exception as err:
            logger.error(f"Error parsing challenge type: {err}")

        # Fallback to visual recognition solution
        return await self.robotic_arm.check_challenge_type()

    async def _solve_captcha(self):
        challenge_type = await self._review_challenge_type()
        logger.debug(
            f"Start Challenge - type={challenge_type.value} count={self.robotic_arm.signal_crumb_count}"
        )

        try:
            # {{< Skip specific challenge questions >}}
            with suppress(Exception):
                if self.config.ignore_request_questions and self._captcha_payload:
                    for q in self.config.ignore_request_questions:
                        if q in self._captcha_payload.get_requester_question():
                            await self.page.wait_for_timeout(2000)
                            await self.robotic_arm.refresh_challenge()
                            return await self._solve_captcha()

            # {{< challenge start >}}
            match challenge_type:
                case RequestType.IMAGE_LABEL_BINARY:
                    if RequestType.IMAGE_LABEL_BINARY not in self.config.ignore_request_types:
                        return await self.robotic_arm.challenge_image_label_binary()
                case challenge_type.IMAGE_LABEL_SINGLE_SELECT:
                    if (
                        RequestType.IMAGE_LABEL_AREA_SELECT not in self.config.ignore_request_types
                        and challenge_type.IMAGE_LABEL_SINGLE_SELECT
                        not in self.config.ignore_request_types
                    ):
                        return await self.robotic_arm.challenge_image_label_select(challenge_type)
                case challenge_type.IMAGE_LABEL_MULTI_SELECT:
                    if (
                        RequestType.IMAGE_LABEL_AREA_SELECT not in self.config.ignore_request_types
                        and challenge_type.IMAGE_LABEL_MULTI_SELECT
                        not in self.config.ignore_request_types
                    ):
                        return await self.robotic_arm.challenge_image_label_select(challenge_type)
                case challenge_type.IMAGE_DRAG_SINGLE:
                    if (
                        RequestType.IMAGE_DRAG_DROP not in self.config.ignore_request_types
                        and ChallengeTypeEnum.IMAGE_DRAG_SINGLE
                        not in self.config.ignore_request_types
                    ):
                        return await self.robotic_arm.challenge_image_drag_drop(challenge_type)
                case challenge_type.IMAGE_DRAG_MULTI:
                    if (
                        RequestType.IMAGE_DRAG_DROP not in self.config.ignore_request_types
                        and ChallengeTypeEnum.IMAGE_DRAG_MULTI
                        not in self.config.ignore_request_types
                    ):
                        return await self.robotic_arm.challenge_image_drag_drop(challenge_type)
                # {{< HCI >}}
                case _:
                    # todo Agentic Workflow | zero-shot challenge
                    logger.warning(f"Unknown types of challenges: {challenge_type}")
            # {{< challenge end >}}

            await self.page.wait_for_timeout(2000)
            await self.robotic_arm.refresh_challenge()
            return await self._solve_captcha()
        except Exception as err:
            # This is an execution error inside the challenge,
            # hcaptcha challenge does not automatically refresh
            logger.exception(f"ChallengeException - type={challenge_type.value} {err=}")
            await self.page.wait_for_timeout(5000)
            await self.robotic_arm.refresh_challenge()
            return await self._solve_captcha()

    async def wait_for_challenge(self) -> ChallengeSignal:
        # Assigning human-computer challenge tasks to the main thread coroutine.
        # ----------------------------------------------------------------------
        try:
            if self._captcha_response_queue.empty():
                await asyncio.wait_for(self._solve_captcha(), timeout=self.config.EXECUTION_TIMEOUT)
        except asyncio.TimeoutError:
            logger.error("Challenge execution timed out", timeout=self.config.EXECUTION_TIMEOUT)
            return ChallengeSignal.EXECUTION_TIMEOUT

        # Waiting for hCAPTCHA response processing result
        # -----------------------------------------------
        # After the completion of the human-machine challenge workflow,
        # it is expected to obtain a signal indicating whether the challenge was successful in the cr_queue.
        logger.debug("Start checking captcha response")
        try:
            cr = await asyncio.wait_for(
                self._captcha_response_queue.get(), timeout=self.config.RESPONSE_TIMEOUT
            )
        except asyncio.TimeoutError:
            logger.error(f"Wait for captcha response timeout {self.config.RESPONSE_TIMEOUT}s")
            return ChallengeSignal.EXECUTION_TIMEOUT
        else:
            # Match: Timeout / Loss
            if not cr or not cr.is_pass:
                if self.config.RETRY_ON_FAILURE:
                    logger.warning("Failed to challenge, try to retry the strategy")
                    await self.page.wait_for_timeout(2000)
                    return await self.wait_for_challenge()
                return ChallengeSignal.FAILURE
            # Match: Success
            if cr.is_pass:
                logger.success("Challenge success")
                self._cache_validated_captcha_response(cr)
                return ChallengeSignal.SUCCESS

        return ChallengeSignal.FAILURE
