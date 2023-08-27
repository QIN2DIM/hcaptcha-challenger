# -*- coding: utf-8 -*-
# Time       : 2023/8/25 14:00
# Author     : QIN2DIM
# GitHub     : https://github.com/QIN2DIM
# Description:
from __future__ import annotations

import random
import re
import time
import warnings
from dataclasses import dataclass
from typing import Tuple

from loguru import logger
from selenium.common.exceptions import (
    WebDriverException,
    TimeoutException,
    NoSuchElementException,
    ElementNotInteractableException,
    ElementNotVisibleException,
    StaleElementReferenceException,
    ElementClickInterceptedException,
    InvalidArgumentException,
)
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.wait import WebDriverWait
from undetected_chromedriver import Chrome

from hcaptcha_challenger.agents.exceptions import ChallengePassed, LabelNotFoundException
from hcaptcha_challenger.agents.skeleton import Skeleton
from hcaptcha_challenger.components.image_downloader import download_images
from hcaptcha_challenger.components.prompt_handler import split_prompt_message, label_cleaning

warnings.filterwarnings("ignore", category=DeprecationWarning)


@dataclass
class SeleniumAgent(Skeleton):
    _threat = 0

    def switch_to_challenge_frame(self, ctx, **kwargs):
        WebDriverWait(ctx, 15, ignored_exceptions=(ElementNotVisibleException,)).until(
            EC.frame_to_be_available_and_switch_to_it(
                (By.XPATH, "//iframe[contains(@src,'#frame=challenge')]")
            )
        )

    def get_label(self, ctx, **kwargs):
        # Scan and determine the type of challenge.
        for _ in range(3):
            try:
                label_obj = WebDriverWait(
                    ctx, 5, ignored_exceptions=(ElementNotVisibleException,)
                ).until(EC.presence_of_element_located((By.XPATH, "//h2[@class='prompt-text']")))
            except TimeoutException:
                raise ChallengePassed("Man-machine challenge unexpectedly passed")
            else:
                self._prompt = label_obj.text
                if self._prompt:
                    break
                time.sleep(1)
                continue
        # Skip the `draw challenge`
        else:
            logger.debug(
                "Pass challenge", challenge="image_label_area_select", site_link=ctx.current_url
            )
            return self.status.CHALLENGE_BACKCALL

        # Continue the `click challenge`
        try:
            _label = split_prompt_message(prompt_message=self._prompt, lang=self.modelhub.lang)
        except (AttributeError, IndexError):
            raise LabelNotFoundException("Get the exception label object")
        else:
            self._label = label_cleaning(_label)
            logger.debug("Get label", name=self._label)

    def mark_samples(self, ctx, *args, **kwargs):
        # 等待图片加载完成
        try:
            WebDriverWait(ctx, 5, ignored_exceptions=(ElementNotVisibleException,)).until(
                EC.presence_of_all_elements_located((By.XPATH, "//div[@class='task-image']"))
            )
        except TimeoutException:
            try:
                ctx.switch_to.default_content()
                WebDriverWait(ctx, 1, 0.1).until(
                    EC.visibility_of_element_located(
                        (By.XPATH, "//div[contains(@class,'hcaptcha-success')]")
                    )
                )
                return self.status.CHALLENGE_SUCCESS
            except WebDriverException:
                return self.status.CHALLENGE_CONTINUE

        time.sleep(0.3)

        # DOM 定位元素
        samples = ctx.find_elements(By.XPATH, "//div[@class='task-image']")
        for sample in samples:
            alias = sample.get_attribute("aria-label")
            while True:
                try:
                    image_style = sample.find_element(By.CLASS_NAME, "image").get_attribute("style")
                    url = re.split(r'[(")]', image_style)[2]
                    self._alias2url.update({alias: url})
                    break
                except IndexError:
                    continue
            self._alias2locator.update({alias: sample})

    def download_images(self):
        container = super().download_images()
        download_images(container=container)

    def challenge(self, ctx, model, *args, **kwargs):
        ta = []
        # {{< IMAGE CLASSIFICATION >}}
        for alias, path in self._alias2path.items():
            t0 = time.time()
            result = model.execute(img_stream=path.read_bytes())
            ta.append(time.time() - t0)
            # Pass: Hit at least one object
            if result:
                try:
                    time.sleep(random.uniform(0.2, 0.3))
                    self._alias2locator[alias].click()
                except StaleElementReferenceException:
                    pass
                except WebDriverException as err:
                    logger.warning(err)

        # {{< SUBMIT ANSWER >}}
        try:
            WebDriverWait(ctx, 15, ignored_exceptions=(ElementClickInterceptedException,)).until(
                EC.element_to_be_clickable((By.XPATH, "//div[@class='button-submit button']"))
            ).click()
        except ElementClickInterceptedException:
            pass
        except WebDriverException as err:
            logger.exception(err)
        logger.debug("Submit challenge", result=f"{self._label}: {round(sum(ta), 2)}s")

    def is_success(self, ctx, *args, **kwargs) -> Tuple[str, str]:
        def is_challenge_image_clickable():
            try:
                WebDriverWait(ctx, 1, poll_frequency=0.1).until(
                    EC.presence_of_element_located((By.XPATH, "//div[@class='task-image']"))
                )
                return True
            except TimeoutException:
                return False

        def is_flagged_flow():
            try:
                WebDriverWait(ctx, 1.2, poll_frequency=0.1).until(
                    EC.visibility_of_element_located((By.XPATH, "//div[@class='error-text']"))
                )
                self._threat += 1
                if self._threat > 3:
                    logger.warning("Your proxy IP may have been flagged")
                return True
            except TimeoutException:
                return False

        time.sleep(1)
        if is_flagged_flow():
            return self.status.CHALLENGE_RETRY, "重置挑战"
        if is_challenge_image_clickable():
            return self.status.CHALLENGE_CONTINUE, "继续挑战"
        return self.status.CHALLENGE_SUCCESS, "退火成功"

    def anti_checkbox(self, ctx, *args, **kwargs):
        for _ in range(8):
            try:
                # [👻] 进入复选框
                WebDriverWait(ctx, 2, ignored_exceptions=(ElementNotVisibleException,)).until(
                    EC.frame_to_be_available_and_switch_to_it(
                        (By.XPATH, "//iframe[contains(@title,'checkbox')]")
                    )
                )
                # [👻] 点击复选框
                WebDriverWait(ctx, 2).until(EC.element_to_be_clickable((By.ID, "checkbox"))).click()
                logger.debug("Handle hCaptcha checkbox")
                return True
            except (TimeoutException, InvalidArgumentException):
                pass
            finally:
                # [👻] 回到主线剧情
                ctx.switch_to.default_content()

    def anti_hcaptcha(self, ctx, *args, **kwargs) -> bool | str:
        # [👻] 它來了！
        try:
            # If it cycles more than twice, your IP has been blacklisted
            for index in range(3):
                # [👻] 進入挑戰框架
                self.switch_to_challenge_frame(ctx)

                # [👻] 獲取挑戰標簽
                if drop := self.get_label(ctx) in [self.status.CHALLENGE_BACKCALL]:
                    ctx.switch_to.default_content()
                    return drop

                # [👻] 編排定位器索引
                if drop := self.mark_samples(ctx) in [
                    self.status.CHALLENGE_SUCCESS,
                    self.status.CHALLENGE_CONTINUE,
                ]:
                    ctx.switch_to.default_content()
                    return drop

                # [👻] 拉取挑戰圖片
                self.download_images()

                # [👻] 滤除无法处理的挑战类别
                if drop := self.tactical_retreat() in [self.status.CHALLENGE_BACKCALL]:
                    ctx.switch_to.default_content()
                    return drop

                # [👻] 注册解决方案
                # 根据挑战类型自动匹配不同的模型
                model = self.match_solution()

                # [👻] 識別|點擊|提交
                self.challenge(ctx, model)

                # [👻] 輪詢控制臺響應
                result, _ = self.is_success(ctx)
                logger.debug("Get response", desc=result)

                ctx.switch_to.default_content()
                if result in [
                    self.status.CHALLENGE_SUCCESS,
                    self.status.CHALLENGE_CRASH,
                    self.status.CHALLENGE_RETRY,
                ]:
                    return result

        except WebDriverException as err:
            logger.exception(err)
            ctx.switch_to.default_content()
            return self.status.CHALLENGE_CRASH


class ArmorUtils:
    @staticmethod
    def face_the_checkbox(ctx: Chrome) -> bool | None:
        try:
            WebDriverWait(ctx, 8, ignored_exceptions=(WebDriverException,)).until(
                EC.presence_of_element_located((By.XPATH, "//iframe[contains(@title,'checkbox')]"))
            )
            return True
        except TimeoutException:
            return False

    @staticmethod
    def get_hcaptcha_response(ctx: Chrome) -> str | None:
        return ctx.execute_script("hcaptcha.getResponse()")

    @staticmethod
    def refresh(ctx: Chrome) -> bool | None:
        try:
            ctx.find_element(By.XPATH, "//div[@class='refresh button']").click()
        except (NoSuchElementException, ElementNotInteractableException):
            return False
        return True
