# -*- coding: utf-8 -*-
# Time       : 2023/8/19 17:17
# Author     : QIN2DIM
# GitHub     : https://github.com/QIN2DIM
# Description:
from __future__ import annotations

import random
import re
import threading
import time
from contextlib import suppress
from dataclasses import dataclass
from typing import Tuple

from loguru import logger
from playwright.sync_api import Error as NinjaError
from playwright.sync_api import FrameLocator, Page
from playwright.sync_api import TimeoutError as NinjaTimeout

from hcaptcha_challenger.agents.exceptions import ChallengePassed
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

    def switch_to_challenge_frame(self, page: Page, window: str = "login", **kwargs):
        if window == "login":
            frame_challenge = page.frame_locator(self.HOOK_CHALLENGE)
        else:
            frame_purchase = page.frame_locator(self.HOOK_PURCHASE)
            frame_challenge = frame_purchase.frame_locator(self.HOOK_CHALLENGE)

        return frame_challenge

    def get_label(self, frame_challenge: FrameLocator, **kwargs):
        try:
            self._prompt = frame_challenge.locator("//h2[@class='prompt-text']").text_content(
                timeout=10000
            )
        except NinjaTimeout:
            raise ChallengePassed("Man-machine challenge unexpectedly passed")

        _label = split_prompt_message(self._prompt, lang="en")
        self._label = label_cleaning(_label)

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
        self, page: Page, frame_challenge: FrameLocator = None, init=True, *args, **kwargs
    ) -> Tuple[str, str]:
        """
        Complex logic for judging the response of a challenge
        :param frame_challenge:
        :param init:
        :param page:
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
            return self.status.CHALLENGE_CONTINUE, "Continue to challenge"
        if is_continue_clickable():
            return self.status.CHALLENGE_CONTINUE, "Continue to challenge"
        return self.status.CHALLENGE_SUCCESS, "退火成功"

    def anti_checkbox(self, page: Page, *args, **kwargs):
        checkbox = page.frame_locator("//iframe[contains(@title,'checkbox')]")
        checkbox.locator("#checkbox").click()

    def anti_hcaptcha(
        self, page: Page, window: str = "login", recur_url=None, *args, **kwargs
    ) -> bool | str:
        frame_challenge = self.switch_to_challenge_frame(page, window)
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
                    logger.debug("Get response", desc=f"{message}({result})")
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
