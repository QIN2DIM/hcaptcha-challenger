# -*- coding: utf-8 -*-
# Time       : 2023/8/19 17:17
# Author     : QIN2DIM
# GitHub     : https://github.com/QIN2DIM
# Description:
from __future__ import annotations

import inspect
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Tuple, Callable, Dict, Any

import httpx
from loguru import logger
from playwright.sync_api import BrowserContext as SyncContext, FrameLocator, Page, sync_playwright
from playwright.sync_api import TimeoutError as NinjaTimeout
from playwright.sync_api import Error as NinjaError

from hcaptcha_challenger.agents.exceptions import ChallengePassed, AuthUnknownException
from hcaptcha_challenger.agents.skeleton import Skeleton
from hcaptcha_challenger.components.image_downloader import download_images
from hcaptcha_challenger.components.prompt_handler import split_prompt_message, label_cleaning


@dataclass
class PlaywrightAgent(Skeleton):
    # //iframe[@id='talon_frame_checkout_free_prod']
    HOOK_PURCHASE = "//div[@id='webPurchaseContainer']//iframe"
    HOOK_CHECKBOX = "//iframe[contains(@title, 'checkbox for hCaptcha')]"
    HOOK_CHALLENGE = "//iframe[contains(@title, 'hCaptcha challenge')]"

    critical_threshold = 3

    def switch_to_challenge_frame(self, ctx, **kwargs):
        pass

    def get_label(self, frame_challenge: FrameLocator, **kwargs):
        try:
            self._prompt = frame_challenge.locator("//h2[@class='prompt-text']").text_content(
                timeout=10000
            )
        except NinjaTimeout:
            raise ChallengePassed("Man-machine challenge unexpectedly passed")

        _label = split_prompt_message(self._prompt, lang="en")
        self._label = label_cleaning(_label)

        if "please click on the" in self._label.lower():
            logger.warning("Pass challenge", label=self._label, case="NotBinaryChallenge")

    def mark_samples(self, frame_challenge: FrameLocator, *args, **kwargs):
        """Get the download link and locator of each challenge image"""
        samples = frame_challenge.locator("//div[@class='task-image']")
        count = samples.count()
        for i in range(count):
            sample = samples.nth(i)
            sample.wait_for()
            alias = sample.get_attribute("aria-label")
            image_style = sample.locator(".image").get_attribute("style")
            url = re.split(r'[(")]', image_style)[2]
            self._alias2url.update({alias: url})
            self._alias2locator.update({alias: sample})

    def download_images(self):
        container = super().download_images()
        t = threading.Thread(target=download_images, kwargs={"container": container})
        t.start()
        t.join()
        # with httpx.Client() as client:
        #     for challenge_img_path, url in container:
        #         data = client.get(url).content
        #         challenge_img_path.write_bytes(data)

    def challenge(self, frame_challenge: FrameLocator, model, *args, **kwargs):
        # {{< IMAGE CLASSIFICATION >}}
        for alias, path in self._alias2path.items():
            result = model.execute(img_stream=path.read_bytes())
            if result:
                with suppress(NinjaTimeout):
                    time.sleep(random.uniform(0.1, 0.3))
                    self._alias2locator[alias].click()

        # {{< SUBMIT ANSWER >}}
        with suppress(NinjaTimeout):
            fl = frame_challenge.locator("//div[@class='button-submit button']")
            fl.click(delay=1000, timeout=5000)

    def is_success(
        self,
        page: Page,
        frame_challenge: FrameLocator = None,
        window=None,
        init=True,
        hook_url=None,
        *args,
        **kwargs,
    ) -> Tuple[str, str]:
        """
        判断挑战是否成功的复杂逻辑
        :param hook_url:
        :param frame_challenge:
        :param init:
        :param window:
        :param page: 挑战者驱动上下文
        :return:
        """

        def is_continue_clickable():
            """ "
            False >>  dom elements hidden
            True >> it's clickable
            """
            try:
                prompts_obj = frame_challenge.locator("//div[@class='error-text']")
                prompts_obj.first.wait_for(timeout=2000)
                logger.debug("Checkout - status=再试一次")
                return True
            except NinjaTimeout:
                task_image = frame_challenge.locator("//div[@class='task-image']")
                task_image.first.wait_for(state="detached", timeout=3000)
                return False
            except NinjaError:
                return False

        def is_init_clickable():
            with suppress(NinjaError):
                return frame_challenge.locator("//div[@class='task-image']").first.is_visible()

        # 首轮测试后判断短时间内页内是否存在可点击的拼图元素
        # hcaptcha 最多两轮验证，一般情况下，账号信息有误仅会执行一轮，然后返回登录窗格提示密码错误
        # 其次是被识别为自动化控制，这种情况也是仅执行一轮，回到登录窗格提示“返回数据错误”
        if init and is_init_clickable():
            return self.status.CHALLENGE_CONTINUE, "继续挑战"
        if is_continue_clickable():
            return self.status.CHALLENGE_CONTINUE, "继续挑战"

        flag = page.url

        if window == "free":
            try:
                page.locator(self.HOOK_PURCHASE).wait_for(state="detached")
                return self.status.CHALLENGE_SUCCESS, "退火成功"
            except NinjaTimeout:
                return self.status.CHALLENGE_RETRY, "決策中斷"
        if window == "login":
            for _ in range(3):
                if hook_url:
                    with suppress(NinjaTimeout):
                        page.wait_for_url(hook_url, timeout=3000)
                        return self.status.CHALLENGE_SUCCESS, "退火成功"
                else:
                    page.wait_for_timeout(2000)
                    if page.url != flag:
                        if "id/login/mfa" not in page.url:
                            return self.status.CHALLENGE_SUCCESS, "退火成功"
                        raise self.status("人机挑战已退出 - error=遭遇意外的 MFA 多重认证")

                mui_typography = page.locator("//h6")
                with suppress(NinjaTimeout):
                    mui_typography.first.wait_for(timeout=1000, state="attached")
                if mui_typography.count() > 1:
                    with suppress(AttributeError):
                        error_text = mui_typography.nth(1).text_content().strip()
                        if "错误回复" in error_text:
                            self.critical_threshold += 1
                            return self.status.CHALLENGE_RETRY, "登入页面错误回复"
                        if "there was a socket open error" in error_text:
                            return self.status.CHALLENGE_RETRY, "there was a socket open error"
                        if self.critical_threshold > 3:
                            logger.debug(f"認證失敗 - {error_text=}")
                            _unknown = AuthUnknownException(msg=error_text)
                            _unknown.report(error_text)
                            raise _unknown

    def anti_checkbox(self, page: Page, *args, **kwargs):
        checkbox = page.frame_locator("//iframe[contains(@title,'checkbox')]")
        checkbox.locator("#checkbox").click()

    def anti_hcaptcha(
        self, page: Page, window: str = "login", recur_url=None, *args, **kwargs
    ) -> bool | str:
        if window == "login":
            frame_challenge = page.frame_locator(self.HOOK_CHALLENGE)
        else:
            frame_purchase = page.frame_locator(self.HOOK_PURCHASE)
            frame_challenge = frame_purchase.frame_locator(self.HOOK_CHALLENGE)

        try:
            # [👻] 人机挑战！
            for i in range(2):
                page.wait_for_timeout(2000)
                # [👻] 获取挑战标签
                self.get_label(frame_challenge)
                # [👻] 編排定位器索引
                self.mark_samples(frame_challenge)
                # [👻] 拉取挑戰圖片
                self.download_images()
                # [👻] 滤除无法处理的挑战类别
                if not self._label_alias.get(self._label):
                    return self.status.CHALLENGE_BACKCALL
                # [👻] 注册解决方案
                # 根据挑战类型自动匹配不同的模型
                model = self.match_solution()
                # [👻] 識別|點擊|提交
                self.challenge(frame_challenge, model=model)
                # [👻] 輪詢控制臺響應
                with suppress(TypeError):
                    result, message = self.is_success(
                        page, frame_challenge, window=window, init=not i, hook_url=recur_url
                    )
                    logger.debug("获取响应", desc=f"{message}({result})")
                    if result in [
                        self.status.CHALLENGE_SUCCESS,
                        self.status.CHALLENGE_CRASH,
                        self.status.CHALLENGE_RETRY,
                    ]:
                        return result
                    page.wait_for_timeout(2000)
        # from::mark_samples url = re.split(r'[(")]', image_style)[2]
        except IndexError:
            return self.anti_hcaptcha(page, window, recur_url)
        except ChallengePassed:
            return self.status.CHALLENGE_SUCCESS
        except Exception as err:
            logger.exception(err)


AgentMan = Callable[[SyncContext], None]
AgentSu = Callable[[SyncContext, ...], None]


class Tarnished:
    def __init__(
        self,
        user_data_dir: Path,
        *,
        record_dir: Path | None = None,
        record_har_path: Path | None = None,
        state_path: Path | None = None,
    ):
        self._user_data_dir = user_data_dir
        self._record_dir = record_dir
        self._record_har_path = record_har_path
        self.state_path = state_path

    @staticmethod
    def _apply_stealth(context: SyncContext):
        enabled_evasions = [
            "chrome.app",
            "chrome.csi",
            "chrome.loadTimes",
            "chrome.runtime",
            "iframe.contentWindow",
            "media.codecs",
            "navigator.hardwareConcurrency",
            "navigator.languages",
            "navigator.permissions",
            "navigator.plugins",
            "navigator.webdriver",
            "sourceurl",
            "webgl.vendor",
            "window.outerdimensions",
        ]

        for e in enabled_evasions:
            evasion_code = (
                Path(__file__)
                .parent.joinpath(f"puppeteer-extra-plugin-stealth/evasions/{e}/index.js")
                .read_text(encoding="utf8")
            )
            context.add_init_script(evasion_code)

        return context

    @staticmethod
    def _patch_cookies(context: SyncContext):
        five_days_ago = datetime.now() - timedelta(days=5)
        cookie = {
            "name": "OptanonAlertBoxClosed",
            "value": five_days_ago.isoformat(),
            "domain": ".epicgames.com",
            "path": "/",
        }
        context.add_cookies([cookie])

    def storage_state(self, context: SyncContext):
        if self.state_path:
            logger.info("Storage ctx_cookie", path=self.state_path)
            context.storage_state(path=self.state_path)

    def execute(self, *, sequence: AgentMan | AgentSu, parameters: Dict[str, Any] = None, **kwargs):
        with sync_playwright() as p:
            context = p.firefox.launch_persistent_context(
                user_data_dir=self._user_data_dir,
                headless=False,
                locale="en-US",
                record_video_dir=self._record_dir,
                record_har_path=self._record_har_path,
                args=["--hide-crash-restore-bubble"],
                **kwargs,
            )
            self._apply_stealth(context)
            self._patch_cookies(context)

            if not isinstance(sequence, list):
                sequence = [sequence]
            for container in sequence:
                logger.info("Execute task", name=container.__name__)
                kws = {}
                params = inspect.signature(container).parameters
                if parameters and isinstance(parameters, dict):
                    for name in params:
                        if name != "context" and name in parameters:
                            kws[name] = parameters[name]
                if not kws:
                    container(context)
                else:
                    container(context, **kws)
            context.close()
