import asyncio
import json
import re
import time
from asyncio import Queue
from contextlib import suppress
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import httpx
import msgpack
from loguru import logger
from playwright.async_api import Page, Response, Locator, TimeoutError, expect
from pydantic import Field, BaseModel

from hcaptcha_challenger.models import RequestType, CaptchaPayload, CaptchaResponse
from hcaptcha_challenger.utils import SiteKey


class CollectorConfig(BaseModel):
    dataset_dir: Path = Path("dataset")

    site_key: str = Field(default=SiteKey.user_easy)

    focus_types: List[RequestType] = Field(
        default_factory=lambda _: [
            RequestType.IMAGE_DRAG_DROP,
            RequestType.IMAGE_LABEL_AREA_SELECT,
            RequestType.IMAGE_LABEL_BINARY,
        ],
        description="Focus on these types of challenges only",
    )

    MAX_LOOP_COUNT: int = Field(default=30, description="Maximum number of loops")
    MAX_RUNNING_TIME: float = Field(default=300, description="Collector single run time (second)")
    WAIT_FOR_TIMEOUT_CHALLENGE_VIEW: float = Field(
        default=2000, description="Waiting for the challenge view to render (millisecond)"
    )


class Collector:
    def __init__(self, page: Page, collector_config: CollectorConfig | None = None):
        self.page = page
        self.config = collector_config or CollectorConfig()

        self._captcha_payload_queue: Queue[CaptchaPayload | None] = Queue()
        self._captcha_response_queue: Queue[CaptchaResponse] = Queue()

        self._loop_control: Queue[int] = Queue()
        self._startup_time = time.time()
        self._current_request_type = None

        self._init_loop_control()

        self.page.on("response", self._task_handler)

    def _init_loop_control(self):
        count = max(self.config.MAX_LOOP_COUNT, 1)
        for _ in range(count):
            self._loop_control.put_nowait(1)

    @property
    def checkbox_selector(self) -> str:
        return "//iframe[starts-with(@src,'https://newassets.hcaptcha.com/captcha/v1/') and contains(@src, 'frame=checkbox')]"

    @property
    def challenge_selector(self) -> str:
        return "//iframe[starts-with(@src,'https://newassets.hcaptcha.com/captcha/v1/') and contains(@src, 'frame=challenge')]"

    @property
    def remaining_progress(self) -> int:
        return self._loop_control.qsize()

    @property
    def current_request_type(self) -> str | None:
        return self._current_request_type

    async def _click_by_mouse(self, locator: Locator):
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

    async def _wake_challenge(self):
        checkbox_frame = self.page.frame_locator(self.checkbox_selector)
        checkbox_element = checkbox_frame.locator("//div[@id='checkbox']")
        await self._click_by_mouse(checkbox_element)

    async def _refresh_challenge(self):
        try:
            refresh_frame = self.page.frame_locator(self.challenge_selector)
            refresh_element = refresh_frame.locator("//div[@class='refresh button']")
            await self._click_by_mouse(refresh_element)
        except TimeoutError as err:
            logger.warning(f"Failed to click refresh button - {err=}")

    async def _wait_for_all_loaders_complete(self):
        """Wait for all loading indicators to complete (become invisible)"""
        frame_challenge = self.page.frame_locator(self.challenge_selector)

        await self.page.wait_for_timeout(self.config.WAIT_FOR_TIMEOUT_CHALLENGE_VIEW)

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
            except Exception as err:
                logger.error(f"Reverse processing getcaptcha failed: {err}")
                self._captcha_payload_queue.put_nowait(None)

    def _create_cache_key(self, captcha_payload: CaptchaPayload) -> Tuple[str, Path]:
        """

        Args:
            captcha_payload:

        Returns: ./dataset / require_type / prompt / current_time

        """
        request_type = captcha_payload.request_type.value
        prompt = captcha_payload.get_requester_question()
        current_datetime = datetime.now()
        current_time = current_datetime.strftime("%Y%m%d/%Y%m%d%H%M%S%f")

        cache_key = self.config.dataset_dir.joinpath(request_type, prompt, current_time)
        crt = current_datetime.strftime("%Y%m%d%H%M%S%f")

        return crt, cache_key

    async def _capture_challenge_view(self, cp: CaptchaPayload, crt: str, cache_key: Path):
        frame_challenge = self.page.frame_locator(self.challenge_selector)

        signal_crumb_count = len(cp.tasklist)
        if cp.request_type == RequestType.IMAGE_LABEL_BINARY:
            signal_crumb_count = int(len(cp.tasklist) / 9)

        for cid in range(signal_crumb_count):
            if cp.request_type == RequestType.IMAGE_LABEL_BINARY:
                await self._wait_for_all_loaders_complete()
            else:
                await self.page.wait_for_timeout(self.config.WAIT_FOR_TIMEOUT_CHALLENGE_VIEW)

            challenge_view = frame_challenge.locator("//div[@class='challenge-view']")
            cache_path = cache_key.joinpath(f"{crt}_{cid}_challenge_view.png")
            await challenge_view.screenshot(type="png", path=cache_path)

            if signal_crumb_count > 1:
                with suppress(TimeoutError):
                    submit_btn = frame_challenge.locator("//div[@class='button-submit button']")
                    await self._click_by_mouse(submit_btn)

    async def _build_dataset(
        self, cp: CaptchaPayload, crt: str, cache_key: Path, client: httpx.AsyncClient
    ):
        if not isinstance(cp, CaptchaPayload):
            return

        self._current_request_type = cp.request_type.value if cp.request_type else "unknown"

        cache_path_captcha = cache_key.joinpath(f"{crt}_captcha.json")
        cache_path_captcha.parent.mkdir(parents=True, exist_ok=True)

        unpacked_data = cp.model_dump(mode="json")
        cache_path_captcha.write_text(
            json.dumps(unpacked_data, indent=2, ensure_ascii=False), encoding="utf8"
        )

        match cp.request_type:
            case RequestType.IMAGE_LABEL_BINARY:
                for j, task in enumerate(cp.tasklist):
                    i = 0 if j <= 8 else 1
                    j = j if j <= 8 else j - 9
                    image_response = await client.get(task.datapoint_uri)
                    cache_path_challenge = cache_key.joinpath(f"{crt}_{i}_{j}_task.png")
                    cache_path_challenge.write_bytes(image_response.content)
                    if cp.requester_question_example:
                        if isinstance(cp.requester_question_example, str):
                            example_response = await client.get(cp.requester_question_example)
                            cache_path_example = cache_key.joinpath(f"{crt}_0_example.png")
                            cache_path_example.write_bytes(example_response.content)
                        elif isinstance(cp.requester_question_example, list):
                            for j, example_uri in enumerate(cp.requester_question_example):
                                example_response = await client.get(example_uri)
                                cache_path_example = cache_key.joinpath(f"{crt}_{j}_example.png")
                                cache_path_example.write_bytes(example_response.content)
            case RequestType.IMAGE_LABEL_AREA_SELECT:
                for i, task in enumerate(cp.tasklist):
                    canvas_response = await client.get(task.datapoint_uri)
                    cache_path_canvas = cache_key.joinpath(f"{crt}_{i}_canvas.png")
                    cache_path_canvas.write_bytes(canvas_response.content)
                if cp.requester_question_example:
                    if isinstance(cp.requester_question_example, str):
                        example_response = await client.get(cp.requester_question_example)
                        cache_path_example = cache_key.joinpath(f"{crt}_0_example.png")
                        cache_path_example.write_bytes(example_response.content)
                    elif isinstance(cp.requester_question_example, list):
                        for j, example_uri in enumerate(cp.requester_question_example):
                            example_response = await client.get(example_uri)
                            cache_path_example = cache_key.joinpath(f"{crt}_{j}_example.png")
                            cache_path_example.write_bytes(example_response.content)
            case RequestType.IMAGE_DRAG_DROP:
                for i, task in enumerate(cp.tasklist):
                    canvas_response = await client.get(task.datapoint_uri)
                    cache_path_canvas = cache_key.joinpath(f"{crt}_{i}_canvas.png")
                    cache_path_canvas.write_bytes(canvas_response.content)
                    for j, entity in enumerate(task.entities):
                        entity_response = await client.get(entity.entity_uri)
                        cache_path_entity = cache_key.joinpath(f"{crt}_{i}_{j}_entity.png")
                        cache_path_entity.write_bytes(entity_response.content)
            case _:
                logger.warning("Unsupported request type")

    @logger.catch
    async def launch(self, *, _by_cli: bool = False):
        _config_log = json.dumps(self.config.model_dump(mode="json"), indent=2, ensure_ascii=False)
        if not _by_cli:
            logger.debug(f"Start collector - {_config_log}")

        if not self.config.focus_types:
            logger.error("No focus types specified")
            return

        site_link = SiteKey.as_site_link(self.config.site_key)
        await self.page.goto(site_link)

        init_status = True

        client = httpx.AsyncClient(http2=True)

        while not self._loop_control.empty():
            # == Update Status == #
            self._loop_control.get_nowait()
            real_running_time = time.time() - self._startup_time
            if real_running_time > self.config.MAX_RUNNING_TIME:
                logger.success(f"Mission ends - running_time={real_running_time:.2f}s")
                return

            # == Wake / Refresh == #
            try:
                if init_status:
                    await self._wake_challenge()
                    init_status = False
                else:
                    await self.page.wait_for_timeout(300)
                    await self._refresh_challenge()
            except Exception as err:
                logger.error(f"Error occurred during challenge: {err}")
                return await self.launch()

            # When clicking on checkbox, the challenge has passed
            if not self._captcha_response_queue.empty():
                self._captcha_response_queue.get_nowait()
                return await self.launch()

            # == Get Captcha == #
            try:
                captcha_payload = await asyncio.wait_for(
                    self._captcha_payload_queue.get(), timeout=10
                )
            except asyncio.TimeoutError:
                logger.error("Wait for captcha payload to timeout")
                continue

            # Download Images
            crt, cache_key = self._create_cache_key(captcha_payload)
            await self._build_dataset(captcha_payload, crt, cache_key, client)
            await self._capture_challenge_view(captcha_payload, crt, cache_key)

            if not _by_cli:
                qsize = self._loop_control.qsize()
                logger.debug(f"Download images - qsize={qsize} type={captcha_payload.request_type}")

        if not _by_cli:
            logger.success(f"Mission ends - loop={self.config.MAX_LOOP_COUNT}")


def check_dataset(captcha_path: Path):
    cp = CaptchaPayload.model_validate_json(captcha_path.read_bytes())
    root = captcha_path.parent

    # 确定信号面包屑数量
    signal_crumb_count = len(cp.tasklist)
    if cp.request_type == RequestType.IMAGE_LABEL_BINARY:
        signal_crumb_count = int(len(cp.tasklist) / 9)

    # 验证challenge_view数量
    cv_paths = list(root.glob("*_challenge_view.png"))
    _verify_file_count(
        actual=len(cv_paths), expected=signal_crumb_count, file_type="challenge_view"
    )

    # 根据请求类型验证不同文件
    if cp.request_type == RequestType.IMAGE_LABEL_BINARY:
        _verify_file_count(
            actual=len(list(root.glob("*_task.png"))), expected=len(cp.tasklist), file_type="task"
        )
    elif cp.request_type in [RequestType.IMAGE_LABEL_AREA_SELECT, RequestType.IMAGE_DRAG_DROP]:
        _verify_file_count(
            actual=len(list(root.glob("*_canvas.png"))),
            expected=len(cp.tasklist),
            file_type="canvas",
        )

        # 仅对DRAG_DROP类型验证entity数量
        if cp.request_type == RequestType.IMAGE_DRAG_DROP:
            for i, task in enumerate(cp.tasklist):
                _verify_file_count(
                    actual=len(list(root.glob(f"*{i}_entity.png"))),
                    expected=len(task.entities),
                    file_type="entity",
                )


def _verify_file_count(actual: int, expected: int, file_type: str):
    """验证文件数量是否符合预期"""
    if actual != expected:
        raise ValueError(f"{file_type} quantity is inaccurate")
