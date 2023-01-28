import asyncio
import os
import random
import re
import sys
import time
import typing
from urllib.parse import quote
from urllib.request import getproxies

import aiohttp
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

from ._solutions import resnet, yolo
from .exceptions import LabelNotFoundException, ChallengePassed, ChallengeLangException


class AshFramework:
    """轻量化的协程控件"""

    def __init__(self, docker: typing.Optional[typing.List] = None):
        if sys.platform.startswith("win") or "cygwin" in sys.platform:
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        else:
            asyncio.set_event_loop(asyncio.new_event_loop())

        # 任务容器：queue
        self.worker, self.done = asyncio.Queue(), asyncio.Queue()
        # 任务容器
        self.docker = docker
        # 任务队列满载时刻长度
        self.max_queue_size = 0

    def progress(self) -> str:
        """任务进度"""
        _progress = self.max_queue_size - self.worker.qsize()
        return f"{_progress}/{self.max_queue_size}"

    def preload(self):
        """预处理"""

    def overload(self):
        """任务重载"""
        if self.docker:
            for task in self.docker:
                self.worker.put_nowait(task)
        self.max_queue_size = self.worker.qsize()

    def offload(self) -> typing.Optional[typing.List]:
        """缓存卸载"""
        crash = []
        while not self.done.empty():
            crash.append(self.done.get())
        return crash

    async def control_driver(self, context, session=None):
        """需要并发执行的代码片段"""
        raise NotImplementedError

    async def launcher(self, session=None):
        """适配接口模式"""
        while not self.worker.empty():
            context = self.worker.get_nowait()
            await self.control_driver(context, session=session)

    async def subvert(self, workers: typing.Union[str, int]):
        """Framework runtime interface"""
        # 任务重载
        self.overload()

        # 弹出空载任务
        if self.max_queue_size == 0:
            return

        # 粘性功率
        workers = self.max_queue_size if workers in ["fast"] else workers
        workers = workers if workers <= self.max_queue_size else self.max_queue_size

        # 弹性分发
        task_list = []
        async with aiohttp.ClientSession() as session:
            for _ in range(workers):
                task = asyncio.create_task(self.launcher(session=session))
                task_list.append(task)
            await asyncio.wait(task_list)

    def perform(self, workers: typing.Union[str, int] = "fast"):
        """Start the highest power coroutine task"""
        asyncio.run(self.subvert(workers))


class HolyChallenger:
    """hCAPTCHA challenge drive control"""

    _label_alias = {
        "zh": {
            "自行车": "bicycle",
            "火车": "train",
            "卡车": "truck",
            "公交车": "bus",
            "巴士": "bus",
            "飞机": "airplane",
            "一条船": "boat",
            "船": "boat",
            "摩托车": "motorcycle",
            "垂直河流": "vertical river",
            "天空中向左飞行的飞机": "airplane in the sky flying left",
            "请选择天空中所有向右飞行的飞机": "airplanes in the sky that are flying to the right",
            "汽车": "car",
            "大象": "elephant",
            "鸟": "bird",
            "狗": "dog",
            "犬科动物": "dog",
            "一匹马": "horse",
            "长颈鹿": "giraffe",
        },
        "en": {
            "airplane": "airplane",
            "motorbus": "bus",
            "bus": "bus",
            "truck": "truck",
            "motorcycle": "motorcycle",
            "boat": "boat",
            "bicycle": "bicycle",
            "train": "train",
            "vertical river": "vertical river",
            "airplane in the sky flying left": "airplane in the sky flying left",
            "Please select all airplanes in the sky that are flying to the right": "airplanes in the sky that are flying to the right",
            "car": "car",
            "elephant": "elephant",
            "bird": "bird",
            "dog": "dog",
            "canine": "dog",
            "horse": "horse",
            "giraffe": "giraffe",
        },
    }

    BAD_CODE = {
        "а": "a",
        "е": "e",
        "e": "e",
        "i": "i",
        "і": "i",
        "ο": "o",
        "с": "c",
        "ԁ": "d",
        "ѕ": "s",
        "һ": "h",
        "у": "y",
        "р": "p",
        "ϳ": "j",
        "ー": "一",
        "土": "士",
    }

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
        dir_workspace: typing.Optional[str] = None,
        lang: typing.Optional[str] = "zh",
        dir_model: typing.Optional[str] = None,
        onnx_prefix: typing.Optional[str] = None,
        screenshot: typing.Optional[bool] = False,
        debug: typing.Optional[bool] = False,
        path_objects_yaml: typing.Optional[str] = None,
        slowdown: typing.Optional[bool] = True,
    ):
        if not isinstance(lang, str) or not self._label_alias.get(lang):
            raise ChallengeLangException(
                f">> ALERT [ArmorCaptcha] Challenge language [{lang}] not yet supported - "
                f"lang={list(self._label_alias.keys())}"
            )

        self.action_name = "ArmorCaptcha"
        self.dir_model = dir_model or os.path.join("datas", "models")
        self.path_objects_yaml = path_objects_yaml or os.path.join("datas", "objects.yaml")
        self.dir_workspace = dir_workspace or os.path.join("datas", "temp_cache", "_challenge")
        self.debug = debug
        self.onnx_prefix = onnx_prefix
        self.screenshot = screenshot
        self.slowdown = slowdown

        # 存储挑战图片的目录
        self.runtime_workspace = ""
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
            path_objects_yaml=self.path_objects_yaml, dir_model=self.dir_model, lang=self.lang
        )
        self.label_alias.update(self.pom_handler.label_alias)

    @property
    def utils(self):
        return ArmorUtils

    @staticmethod
    def split_prompt_message(prompt_message: str, lang: str) -> str:
        """Detach label from challenge prompt"""
        if lang.startswith("zh"):
            if "中包含" in prompt_message or "上包含" in prompt_message:
                return re.split(r"击|(的每)", prompt_message)[2]
            if "的每" in prompt_message:
                return re.split(r"(包含)|(的每)", prompt_message)[3]
            if "包含" in prompt_message:
                return re.split(r"(包含)|(的图)", prompt_message)[3]
        elif lang.startswith("en"):
            prompt_message = prompt_message.replace(".", "").lower()
            if "containing" in prompt_message:
                th = re.split(r"containing", prompt_message)[-1][1:].strip()
                return th[2:].strip() if th.startswith("a") else th
            if "select all" in prompt_message:
                return re.split(r"all (.*) images", prompt_message)[1].strip()
        return prompt_message

    def label_cleaning(self, raw_label: str) -> str:
        """cleaning errors-unicode"""
        clean_label = raw_label
        for c in self.BAD_CODE:
            clean_label = clean_label.replace(c, self.BAD_CODE[c])
        return clean_label

    def _init_workspace(self):
        """初始化工作目录，存放缓存的挑战图片"""
        _prefix = (
            f"{time.time()}" + f"_{self.label_alias.get(self.label, '')}" if self.label else ""
        )
        _workspace = os.path.join(self.dir_workspace, _prefix)
        os.makedirs(_workspace, exist_ok=True)
        return _workspace

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
        _out_dir = os.path.join(os.path.dirname(self.dir_workspace), "captcha_screenshot")
        _out_path = os.path.join(_out_dir, _filename)
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

    def log(
        self, message: str, _reporter: typing.Optional[bool] = False, **params
    ) -> typing.Optional[str]:
        """格式化日志信息"""
        if not self.debug:
            return

        motive = "Challenge"
        flag_ = f">> {motive} [{self.action_name}] {message}"
        if params:
            flag_ += " - "
            flag_ += " ".join([f"{i[0]}={i[1]}" for i in params.items()])
        if _reporter is True:
            return flag_
        logger.debug(flag_)

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
            self.log(
                message="Pass challenge",
                challenge="image_label_area_select",
                site_link=ctx.current_url,
                screenshot=self.captcha_screenshot(ctx, fn),
            )
            return self.CHALLENGE_BACKCALL

        # Continue the `click challenge`
        try:
            _label = self.split_prompt_message(prompt_message=self.prompt, lang=self.lang)
        except (AttributeError, IndexError):
            raise LabelNotFoundException("Get the exception label object")
        else:
            self.label = self.label_cleaning(_label)
            self.log(message="Get label", label=f"「{self.label}」")

    def tactical_retreat(self, ctx) -> typing.Optional[str]:
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
                f">> ALERT [{self.action_name}] Types of challenges not yet scheduled - "
                f"label=「{self.label}」 "
                f"prompt=「{self.prompt}」 "
                f"screenshot={self.path_screenshot} "
                f"site_link={ctx.current_url} "
                f"issues=https://github.com/QIN2DIM/hcaptcha-challenger/issues?q={q}"
            )
            return self.CHALLENGE_BACKCALL

    def switch_solution(self):
        """Optimizing solutions based on different challenge labels"""
        label_alias = self.label_alias.get(self.label)

        # Load ONNX model - ResNet | YOLO
        if label_alias not in self.pom_handler.fingers:
            self.log("lazy-loading", sign="YOLO", match=label_alias)
            return yolo.YOLO(self.dir_model, self.onnx_prefix)
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
        """
        Download Challenge Image

        ### hcaptcha has a challenge duration limit

        If the page element is not manipulated for a period of time,
        the <iframe> box will disappear and the previously acquired Element Locator will be out of date.
        Need to use some modern methods to shorten the time of `getting the dataset` as much as possible.

        ### Solution

        1. Coroutine Downloader
          Use the coroutine-based method to _pull the image to the local, the best practice (this method).
          In the case of poor network, _pull efficiency is at least 10 times faster than traversal download.

        2. Screen cut
          There is some difficulty in coding.
          Directly intercept nine pictures of the target area, and use the tool function to cut and identify them.
          Need to weave the locator index yourself.

        :return:
        """

        class ImageDownloader(AshFramework):
            """Coroutine Booster - Improve the download efficiency of challenge images"""

            http_proxy: typing.Optional[str] = getproxies().get("http")

            async def control_driver(self, context, session=None):
                path_challenge_img, url = context

                # Download Challenge Image
                async with session.get(url, proxy=self.http_proxy) as response:
                    with open(path_challenge_img, "wb") as file:
                        file.write(await response.read())

        # Initialize the challenge image download directory
        self.runtime_workspace = self._init_workspace()

        # Initialize the data container
        docker_ = []
        for alias_, url_ in self.alias2url.items():
            path_challenge_img_ = os.path.join(self.runtime_workspace, f"{alias_}.png")
            self.alias2path.update({alias_: path_challenge_img_})
            docker_.append((path_challenge_img_, url_))

        # Initialize the coroutine-based image downloader
        start = time.time()
        ImageDownloader(docker_).perform()
        self.log(message="Download challenge images", timeit=f"{round(time.time() - start, 2)}s")

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

        self.log(message=f"Submit the challenge - {model.flag}: {round(sum(ta), 2)}s")

    def challenge_success(self, ctx: Chrome) -> typing.Tuple[str, str]:
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
                    logger.warning(f"Your proxy IP may have been flagged - proxies={getproxies()}")
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
                self.log("Handle hCaptcha checkbox")
                return True
            except (TimeoutException, InvalidArgumentException):
                pass
            finally:
                # [👻] 回到主线剧情
                ctx.switch_to.default_content()

    def anti_hcaptcha(self, ctx: Chrome) -> typing.Union[bool, str]:
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
                self.log("Get response", desc=result)

                ctx.switch_to.default_content()
                solution.offload()
                if result in [self.CHALLENGE_SUCCESS, self.CHALLENGE_CRASH, self.CHALLENGE_RETRY]:
                    return result

        except WebDriverException as err:
            logger.exception(err)
            ctx.switch_to.default_content()
            return self.CHALLENGE_CRASH

    def classify(
        self, prompt: str, images: typing.List[typing.Union[str, bytes]]
    ) -> typing.Optional[typing.List[bool]]:
        """TaskType: HcaptchaClassification"""
        if not prompt or not isinstance(prompt, str) or not images or not isinstance(images, list):
            logger.error(
                f">> ALERT [{self.action_name}] Invalid parameters - "
                f"prompt=「{self.prompt}」 "
                f"images=「{images}」"
            )
            return

        self.lang = "zh" if re.compile("[\u4e00-\u9fa5]+").search(prompt) else "en"
        self.label_alias = self._label_alias[self.lang]
        self.label_alias.update(self.pom_handler.get_label_alias(self.lang))
        self.prompt = prompt
        _label = self.split_prompt_message(prompt, lang=self.lang)
        self.label = self.label_cleaning(_label)

        if self.label not in self.label_alias:
            logger.error(
                f">> ALERT [{self.action_name}] Types of challenges not yet scheduled - "
                f"label=「{self.label}」 "
                f"prompt=「{self.prompt}」"
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
    def face_the_checkbox(ctx: Chrome) -> typing.Optional[bool]:
        try:
            WebDriverWait(ctx, 8, ignored_exceptions=(WebDriverException,)).until(
                EC.presence_of_element_located((By.XPATH, "//iframe[contains(@title,'checkbox')]"))
            )
            return True
        except TimeoutException:
            return False

    @staticmethod
    def get_hcaptcha_response(ctx: Chrome) -> typing.Optional[str]:
        return ctx.execute_script("return hcaptcha.getResponse()")

    @staticmethod
    def refresh(ctx: Chrome) -> typing.Optional[bool]:
        try:
            ctx.find_element(By.XPATH, "//div[@class='refresh button']").click()
        except (NoSuchElementException, ElementNotInteractableException):
            return False
        return True
