from __future__ import annotations

import os
import random
import re
import time
from pathlib import Path
from typing import Tuple, List
from urllib.parse import quote
from urllib.request import getproxies

from loguru import logger
from selenium.common.exceptions import (
    ElementNotVisibleException,
    ElementClickInterceptedException,
    WebDriverException,
    TimeoutException,
    NoSuchElementException,
    StaleElementReferenceException,
    ElementNotInteractableException,
    InvalidArgumentException,
)
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.wait import WebDriverWait
from undetected_chromedriver import Chrome

from hcaptcha_challenger.agents.exceptions import (
    LabelNotFoundException,
    ChallengePassed,
    ChallengeLangException,
)
from hcaptcha_challenger.components.image_downloader import download_images
from hcaptcha_challenger.components.prompt_handler import label_cleaning, split_prompt_message
from hcaptcha_challenger.onnx import resnet


class HolyChallenger:
    """hCAPTCHA challenge drive control"""

    _label_alias = {"zh": {}, "en": {}}

    HOOK_CHALLENGE = "//iframe[contains(@src,'#frame=challenge')]"

    # <success> Challenge Passed by following the expected
    CHALLENGE_SUCCESS = "success"
    # <continue> Continue the challenge
    CHALLENGE_CONTINUE = "continue"
    # <crash> Failure of the challenge as expected
    CHALLENGE_CRASH = "crash"
    # <retry> Your proxy IP may have been flagged
    CHALLENGE_RETRY = "retry"
    # <refresh> Skip the specified label as expected
    CHALLENGE_REFRESH = "refresh"
    # <backcall> (New Challenge) Types of challenges not yet scheduled
    CHALLENGE_BACKCALL = "backcall"

    def __init__(
        self,
        dir_workspace: Path,
        models_dir: Path,
        objects_path: Path,
        lang: str | None = "zh",
        onnx_prefix: str | None = None,
        screenshot: bool | None = False,
        debug: bool | None = False,
        slowdown: bool | None = True,
    ):
        if not isinstance(lang, str) or not self._label_alias.get(lang):
            raise ChallengeLangException(
                f">> ALERT [ArmorCaptcha] Challenge language [{lang}] not yet supported - "
                f"lang={list(self._label_alias.keys())}"
            )

        self.action_name = "ArmorCaptcha"
        self.models_dir = models_dir
        self.objects_path = objects_path
        self.dir_workspace = dir_workspace
        self.debug = debug
        self.onnx_prefix = onnx_prefix
        self.screenshot = screenshot
        self.slowdown = slowdown

        # 挑战截图存储路径
        self.path_screenshot = ""
        # 博大精深！
        self.lang = lang
        self.label_alias: dict = self._label_alias[lang]

        # Store the `element locator` of challenge images {挑战图片1: locator1, ...}
        self.alias2locator = {}
        # Store the `download link` of the challenge image {挑战图片1: url1, ...}
        self.alias2url = {}
        # Store the `directory` of challenge image {挑战图片1: "/images/挑战图片1.png", ...}
        self.alias2path = {}
        # 图像标签
        self.label = ""
        self.prompt = ""

        self.threat = 0

        # Automatic registration
        self.pom_handler = resnet.PluggableONNXModels(
            path_objects_yaml=self.objects_path, dir_model=self.models_dir, lang=self.lang
        )
        self.label_alias.update(self.pom_handler.label_alias)

    @property
    def utils(self):
        return ArmorUtils

    def captcha_screenshot(self, ctx, name_screenshot: str = None):
        """
        保存挑战截图，需要在 get_label 之后执行

        :param name_screenshot: filename of the Challenge image
        :param ctx: Webdriver 或 Element
        :return:
        """
        _suffix = self.label_alias.get(self.label, self.label)
        _filename = (
            f"{int(time.time())}.{_suffix}.png" if name_screenshot is None else name_screenshot
        )
        _out_dir = self.dir_workspace.parent.joinpath("captcha_screenshot")
        _out_path = _out_dir.joinpath(_filename)
        os.makedirs(_out_dir, exist_ok=True)

        # FullWindow screenshot or FocusElement screenshot
        try:
            ctx.screenshot(_out_path)
        except AttributeError:
            ctx.save_screenshot(_out_path)
        except Exception as err:
            logger.exception(err)
        finally:
            return _out_path

    def switch_to_challenge_frame(self, ctx: Chrome):
        WebDriverWait(ctx, 15, ignored_exceptions=(ElementNotVisibleException,)).until(
            EC.frame_to_be_available_and_switch_to_it((By.XPATH, self.HOOK_CHALLENGE))
        )

    def get_label(self, ctx: Chrome):
        """
        获取人机挑战需要识别的图片类型（标签）

        :param ctx:
        :return:
        """

        # Scan and determine the type of challenge.
        for _ in range(3):
            try:
                label_obj = WebDriverWait(
                    ctx, 5, ignored_exceptions=(ElementNotVisibleException,)
                ).until(EC.presence_of_element_located((By.XPATH, "//h2[@class='prompt-text']")))
            except TimeoutException:
                raise ChallengePassed("Man-machine challenge unexpectedly passed")
            else:
                self.prompt = label_obj.text
                if self.prompt:
                    break
                time.sleep(1)
                continue
        # Skip the `draw challenge`
        else:
            fn = f"{int(time.time())}.image_label_area_select.png"
            logger.debug(
                "Pass challenge",
                challenge="image_label_area_select",
                site_link=ctx.current_url,
                screenshot=self.captcha_screenshot(ctx, fn),
            )
            return self.CHALLENGE_BACKCALL

        # Continue the `click challenge`
        try:
            _label = split_prompt_message(prompt_message=self.prompt, lang=self.lang)
        except (AttributeError, IndexError):
            raise LabelNotFoundException("Get the exception label object")
        else:
            self.label = label_cleaning(_label)
            logger.debug("Get label", name=self.label)

    def tactical_retreat(self, ctx) -> str | None:
        """
        「blacklist mode」 skip unchoreographed challenges
        :param ctx:
        :return: the screenshot storage path
        """
        if self.label_alias.get(self.label):
            return self.CHALLENGE_CONTINUE

        # Save a screenshot of the challenge
        try:
            challenge_container = ctx.find_element(By.XPATH, "//body[@class='no-selection']")
            self.path_screenshot = self.captcha_screenshot(challenge_container)
        except NoSuchElementException:
            pass
        except WebDriverException as err:
            logger.exception(err)
        finally:
            q = quote(self.label, "utf8")
            logger.warning(
                "Types of challenges not yet scheduled",
                label=self.label,
                prompt=self.prompt,
                shot=f"{self.path_screenshot}",
                site_link=ctx.current_url,
                issue=f"https://github.com/QIN2DIM/hcaptcha-challenger/issues?q={q}",
            )
            return self.CHALLENGE_BACKCALL

    def switch_solution(self):
        """Optimizing solutions based on different challenge labels"""
        label_alias = self.label_alias.get(self.label)

        # Load ONNX model - ResNet | YOLO
        return self.pom_handler.lazy_loading(label_alias)

    def mark_samples(self, ctx: Chrome):
        """
        Get the download link and locator of each challenge image

        :param ctx:
        :return:
        """
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
                return self.CHALLENGE_SUCCESS
            except WebDriverException:
                return self.CHALLENGE_CONTINUE

        time.sleep(0.3)

        # DOM 定位元素
        samples = ctx.find_elements(By.XPATH, "//div[@class='task-image']")
        for sample in samples:
            alias = sample.get_attribute("aria-label")
            while True:
                try:
                    image_style = sample.find_element(By.CLASS_NAME, "image").get_attribute("style")
                    url = re.split(r'[(")]', image_style)[2]
                    self.alias2url.update({alias: url})
                    break
                except IndexError:
                    continue
            self.alias2locator.update({alias: sample})

    def download_images(self):
        prefix = ""
        if self.label:
            prefix = f"{time.time()}_{self.label_alias.get(self.label, '')}"
        runtime_dir = self.dir_workspace.joinpath(prefix)
        runtime_dir.mkdir(mode=777, parents=True, exist_ok=True)

        # Initialize the data container
        container = []
        for alias_, url_ in self.alias2url.items():
            challenge_img_path = runtime_dir.joinpath(f"{alias_}.png")
            self.alias2path.update({alias_: challenge_img_path})
            container.append((challenge_img_path, url_))

        # Initialize the coroutine-based image downloader
        download_images(container)

    def challenge(self, ctx: Chrome, model):
        """
        图像分类，元素点击，答案提交

        ### 性能瓶颈

        此部分图像分类基于 CPU 运行。如果服务器资源极其紧张，图像分类任务可能无法按时完成。
        根据实验结论来看，如果运行时内存少于 512MB，且仅有一个逻辑线程的话，基本上是与深度学习无缘了。

        ### 优雅永不过时

        `hCaptcha` 的挑战难度与 `reCaptcha v2` 不在一个级别。
        这里只要正确率上去就行，也即正确图片覆盖更多，通过率越高（即使因此多点了几个干扰项也无妨）。
        所以这里要将置信度尽可能地调低（未经针对训练的模型本来就是用来猜的）。

        :return:
        """

        ta = []
        # {{< IMAGE CLASSIFICATION >}}
        for alias in self.alias2path:
            # Read binary data weave into types acceptable to the model
            with open(self.alias2path[alias], "rb") as file:
                data = file.read()
            # Get detection results
            t0 = time.time()
            result = model.solution(img_stream=data, label=self.label_alias[self.label])
            ta.append(time.time() - t0)
            # Pass: Hit at least one object
            if result:
                try:
                    # Add a short sleep so that the user
                    # can see the prediction results of the model
                    if self.slowdown:
                        time.sleep(random.uniform(0.2, 0.3))
                    self.alias2locator[alias].click()
                except StaleElementReferenceException:
                    pass
                except WebDriverException as err:
                    logger.warning(err)

        # Check result of the challenge.
        if self.screenshot:
            _filename = f"{int(time.time())}.{model.flag}.{self.label_alias[self.label]}.png"
            self.captcha_screenshot(ctx, name_screenshot=_filename)

        # {{< SUBMIT ANSWER >}}
        try:
            WebDriverWait(ctx, 15, ignored_exceptions=(ElementClickInterceptedException,)).until(
                EC.element_to_be_clickable((By.XPATH, "//div[@class='button-submit button']"))
            ).click()
        except ElementClickInterceptedException:
            pass
        except WebDriverException as err:
            logger.exception(err)
        logger.debug("Submit challenge", result=f"{model.flag}: {round(sum(ta), 2)}s")

    def challenge_success(self, ctx: Chrome) -> Tuple[str, str]:
        """
        判断挑战是否成功的复杂逻辑

        # 首轮测试后判断短时间内页内是否存在可点击的拼图元素
        # hcaptcha 最多两轮验证，一般情况下，账号信息有误仅会执行一轮，然后返回登录窗格提示密码错误
        # 其次是被识别为自动化控制，这种情况也是仅执行一轮，回到登录窗格提示“返回数据错误”

        经过首轮识别点击后，出现四种结果:
            1. 直接通过验证（小概率）
            2. 进入第二轮（正常情况）
                通过短时间内可否继续点击拼图来断言是否陷入第二轮测试
            3. 要求重试（小概率）
                特征被识别|网络波动|被标记的（代理）IP
            4. 通过验证，弹出 2FA 双重认证
              无法处理，任务结束

        :param ctx: 挑战者驱动上下文
        :return:
        """

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
                self.threat += 1
                if getproxies() and self.threat > 3:
                    logger.warning("Your proxy IP may have been flagged", proxies=getproxies())
                return True
            except TimeoutException:
                return False

        time.sleep(1)
        if is_flagged_flow():
            return self.CHALLENGE_RETRY, "重置挑战"
        if is_challenge_image_clickable():
            return self.CHALLENGE_CONTINUE, "继续挑战"
        return self.CHALLENGE_SUCCESS, "退火成功"

    def anti_checkbox(self, ctx: Chrome):
        """处理复选框"""
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

    def anti_hcaptcha(self, ctx: Chrome) -> bool | str:
        """
        Handle hcaptcha challenge

        ## Method

        具体思路是：
        1. 进入 hcaptcha iframe
        2. 获取图像标签
            需要加入判断，有时候 `hcaptcha` 计算的威胁程度极低，会直接让你过，
            于是图像标签之类的元素都不会加载在网页上。
        3. 获取各个挑战图片的下载链接及网页元素位置
        4. 图片下载，分类
            需要用一些技术手段缩短这部分操作的耗时。人机挑战有时间限制。
        5. 对正确的图片进行点击
        6. 提交答案
        7. 判断挑战是否成功
            一般情况下 `hcaptcha` 的验证有两轮，
            而 `recaptcha vc2` 之类的人机挑战就说不准了，可能程序一晚上都在“循环”。

        ## Reference

        M. I. Hossen and X. Hei, "A Low-Cost Attack against the hCaptcha System," 2021 IEEE Security
        and Privacy Workshops (SPW), 2021, pp. 422-431, doi: 10.1109/SPW53761.2021.00061.

        > ps:该篇文章中的部分内容已过时，如今的 hcaptcha challenge 远没有作者说的那么容易应付。
        :param ctx:
        :return:
        """

        # [👻] 它來了！
        try:
            # If it cycles more than twice, your IP has been blacklisted
            for index in range(3):
                # [👻] 進入挑戰框架
                self.switch_to_challenge_frame(ctx)

                # [👻] 獲取挑戰標簽
                if drop := self.get_label(ctx) in [self.CHALLENGE_BACKCALL]:
                    ctx.switch_to.default_content()
                    return drop

                # [👻] 編排定位器索引
                if drop := self.mark_samples(ctx) in [
                    self.CHALLENGE_SUCCESS,
                    self.CHALLENGE_CONTINUE,
                ]:
                    ctx.switch_to.default_content()
                    return drop

                # [👻] 拉取挑戰圖片
                self.download_images()

                # [👻] 滤除无法处理的挑战类别
                if drop := self.tactical_retreat(ctx) in [self.CHALLENGE_BACKCALL]:
                    ctx.switch_to.default_content()
                    return drop

                # [👻] 注册解决方案
                # 根据挑战类型自动匹配不同的模型
                solution = self.switch_solution()

                # [👻] 識別|點擊|提交
                self.challenge(ctx, solution)

                # [👻] 輪詢控制臺響應
                result, _ = self.challenge_success(ctx)
                logger.debug("Get response", desc=result)

                ctx.switch_to.default_content()
                solution.offload()
                if result in [self.CHALLENGE_SUCCESS, self.CHALLENGE_CRASH, self.CHALLENGE_RETRY]:
                    return result

        except WebDriverException as err:
            logger.exception(err)
            ctx.switch_to.default_content()
            return self.CHALLENGE_CRASH

    def classify(self, prompt: str, images: List[str | bytes]) -> List[bool] | None:
        """TaskType: HcaptchaClassification"""
        if not prompt or not isinstance(prompt, str) or not images or not isinstance(images, list):
            logger.error(
                "Invalid parameters", action=self.action_name, prompt=self.prompt, images=images
            )
            return

        self.lang = "zh" if re.compile("[\u4e00-\u9fa5]+").search(prompt) else "en"
        self.label_alias = self._label_alias[self.lang]
        self.label_alias.update(self.pom_handler.get_label_alias(self.lang))
        self.prompt = prompt
        _label = split_prompt_message(prompt, lang=self.lang)
        self.label = label_cleaning(_label)

        if self.label not in self.label_alias:
            logger.error(
                "Types of challenges not yet scheduled", label=self.label, prompt=self.prompt
            )
            return

        model = self.switch_solution()
        response = []
        for img in images:
            try:
                if isinstance(img, str) and os.path.isfile(img):
                    with open(img, "rb") as file:
                        response.append(
                            model.solution(
                                img_stream=file.read(), label=self.label_alias[self.label]
                            )
                        )
                elif isinstance(img, bytes):
                    response.append(
                        model.solution(img_stream=img, label=self.label_alias[self.label])
                    )
                else:
                    response.append(False)
            except Exception as err:
                logger.exception(err)
                response.append(False)
        return response


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
        return ctx.execute_script("return hcaptcha.getResponse()")

    @staticmethod
    def refresh(ctx: Chrome) -> bool | None:
        try:
            ctx.find_element(By.XPATH, "//div[@class='refresh button']").click()
        except (NoSuchElementException, ElementNotInteractableException):
            return False
        return True
