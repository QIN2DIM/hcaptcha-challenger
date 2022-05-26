import asyncio
import os
import re
import sys
import time
import urllib.request
from typing import Optional, Union, Tuple

from selenium.common.exceptions import (
    ElementNotVisibleException,
    ElementClickInterceptedException,
    WebDriverException,
    TimeoutException,
)
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.wait import WebDriverWait
from undetected_chromedriver import Chrome

from services.settings import logger, PATH_RAINBOW
from services.utils import AshFramework
from .exceptions import (
    LabelNotFoundException,
    ChallengePassed,
    ChallengeTimeout,
    AssertTimeout,
    ChallengeLangException,
)
from .solutions import sk_recognition, resnet, yolo


class ArmorCaptcha:
    """hCAPTCHA challenge drive control"""

    label_alias = {
        "zh": {
            "水上飞机": "seaplane",
            "自行车": "bicycle",
            "火车": "train",
            "卡车": "truck",
            "公交车": "bus",
            "巴土": "bus",
            "巴士": "bus",
            "飞机": "aeroplane",
            "ー条船": "boat",
            "船": "boat",
            "摩托车": "motorbike",
            "垂直河流": "vertical river",
            "天空中向左飞行的飞机": "airplane in the sky flying left",
            "请选择天空中所有向右飞行的飞机": "airplanes in the sky that are flying to the right",
            "请选择所有用树叶画的大象": "elephants drawn with leaves",
        },
        "en": {
            "seaplane": "seaplane",
            "ѕeaplane": "seaplane",
            "airplane": "aeroplane",
            "аirplane": "aeroplane",
            "motorbus": "bus",
            "mοtorbus": "bus",
            "bus": "bus",
            "truck": "truck",
            "truсk": "truck",
            "motorcycle": "motorbike",
            "mοtorcycle": "motorbike",
            "boat": "boat",
            "bicycle": "bicycle",
            "train": "train",
            "trаin": "train",
            "vertical river": "vertical river",
            "airplane in the sky flying left": "airplane in the sky flying left",
            "Please select all airplanes in the sky that are flying to the rіght": "airplanes in the sky that are flying to the right",
            "Please select all airplanes in the sky that are flying to the right": "airplanes in the sky that are flying to the right",
            "Please select all the elephants drawn with lеaves": "elephants drawn with leaves",
            "Please select all the elephants drawn with leaves": "elephants drawn with leaves",
        },
    }

    HOOK_CHALLENGE = "//iframe[contains(@title,'content')]"

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

    def __init__(self, dir_workspace: str = None, lang: Optional[str] = "zh", debug=False):
        if not isinstance(lang, str) or not self.label_alias.get(lang):
            raise ChallengeLangException(
                f"Challenge language [{lang}] not yet supported."
                f" -lang={list(self.label_alias.keys())}"
            )

        self.action_name = "ArmorCaptcha"
        self.debug = debug

        # 存储挑战图片的目录
        self.runtime_workspace = ""

        # 博大精深！
        self.lang = lang
        self.label_alias: dict = self.label_alias[lang]

        # Store the `element locator` of challenge images {挑战图片1: locator1, ...}
        self.alias2locator = {}
        # Store the `download link` of the challenge image {挑战图片1: url1, ...}
        self.alias2url = {}
        # Store the `directory` of challenge image {挑战图片1: "/images/挑战图片1.png", ...}
        self.alias2path = {}
        # 图像标签
        self.label = ""
        # 运行缓存
        self.dir_workspace = dir_workspace if dir_workspace else "."

        self.threat = 0

    def _init_workspace(self):
        """初始化工作目录，存放缓存的挑战图片"""
        _prefix = (
            f"{int(time.time())}" + f"_{self.label_alias.get(self.label, '')}" if self.label else ""
        )
        _workspace = os.path.join(self.dir_workspace, _prefix)
        if not os.path.exists(_workspace):
            os.mkdir(_workspace)
        return _workspace

    def log(self, message: str, **params) -> None:
        """格式化日志信息"""
        if not self.debug:
            return

        motive = "Challenge"
        flag_ = f">> {motive} [{self.action_name}] {message}"
        if params:
            flag_ += " - "
            flag_ += " ".join([f"{i[0]}={i[1]}" for i in params.items()])
        logger.debug(flag_)

    def switch_to_challenge_frame(self, ctx: Chrome):
        WebDriverWait(ctx, 15, ignored_exceptions=ElementNotVisibleException).until(
            EC.frame_to_be_available_and_switch_to_it((By.XPATH, self.HOOK_CHALLENGE))
        )

    def split_prompt_message(self, prompt_message: str) -> str:
        """根据指定的语种在提示信息中分离挑战标签"""
        labels_mirror = {
            "zh": re.split(r"[包含 图片]", prompt_message)[2][:-1]
            if "包含" in prompt_message
            else prompt_message,
            "en": re.split(r"containing a", prompt_message)[-1][1:].strip()
            if "containing" in prompt_message
            else prompt_message,
        }
        return labels_mirror[self.lang]

    def get_label(self, ctx: Chrome):
        """
        获取人机挑战需要识别的图片类型（标签）

        :param ctx:
        :return:
        """
        # Necessary.
        time.sleep(0.5)

        try:
            label_obj = WebDriverWait(ctx, 5, ignored_exceptions=ElementNotVisibleException).until(
                EC.presence_of_element_located((By.XPATH, "//div[@class='prompt-text']"))
            )
        except TimeoutException:
            raise ChallengePassed("人机挑战意外通过")

        try:
            _label = self.split_prompt_message(prompt_message=label_obj.text)
        except (AttributeError, IndexError):
            raise LabelNotFoundException("获取到异常的标签对象。")
        else:
            self.label = _label
            if self.label_alias.get(self.label):
                self.log(message="Get label", label=f"「{self.label}」")

    def tactical_retreat(self) -> Optional[str]:
        """模型存在泛化死角，遇到指定标签时主动进入下一轮挑战，节约时间"""
        retreat_labels = []

        pending_label = self.label_alias.get(self.label)

        if not pending_label:
            self.log(message="Types of challenges not yet scheduled", prompt=f"「{self.label}」")
            return self.CHALLENGE_BACKCALL
        if pending_label in retreat_labels:
            self.log(message="Avoiding the unmanageable challenge", label=f"「{self.label}」")
            return self.CHALLENGE_REFRESH
        return self.CHALLENGE_CONTINUE

    def switch_solution(self, dir_model, onnx_prefix):
        """模型卸载"""
        label = self.label_alias.get(self.label)
        if label in ["seaplane"]:
            return resnet.ResNetSeaplane(dir_model)
        if label in ["elephants drawn with leaves"]:
            return resnet.ElephantsDrawnWithLeaves(dir_model, path_rainbow=PATH_RAINBOW)
        if label in ["vertical river"]:
            return sk_recognition.VerticalRiverRecognition(path_rainbow=PATH_RAINBOW)
        if label in ["airplane in the sky flying left"]:
            return sk_recognition.LeftPlaneRecognition(path_rainbow=PATH_RAINBOW)
        if label in ["airplanes in the sky that are flying to the right"]:
            return sk_recognition.RightPlaneRecognition(path_rainbow=PATH_RAINBOW)
        if label in ["horses drawn with flowers"]:
            return resnet.HorsesDrawnWithFlowers(dir_model, path_rainbow=PATH_RAINBOW)
        return yolo.YOLO(dir_model, onnx_prefix)

    def mark_samples(self, ctx: Chrome):
        """
        获取每个挑战图片的下载链接以及网页元素位置

        :param ctx:
        :return:
        """
        # self.log(message="Get challenge image links and element locators")

        # 等待图片加载完成
        WebDriverWait(ctx, 10, ignored_exceptions=ElementNotVisibleException).until(
            EC.presence_of_all_elements_located((By.XPATH, "//div[@class='task-image']"))
        )
        time.sleep(1)

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
        下载挑战图片

        ### hcaptcha 设有挑战时长的限制

          如果一段时间内没有操作页面元素，<iframe> 框体就会消失，之前获取的 Element Locator 将过时。
          需要借助一些现代化的方法尽可能地缩短 `获取数据集` 的耗时。

        ### 解决方案

        1. 使用基于协程的方法拉取图片到本地，最佳实践（本方法）。拉取效率比遍历下载提升至少 10 倍。
        2. 截屏切割，有一定的编码难度。直接截取目标区域的九张图片，使用工具函数切割后识别。需要自己编织定位器索引。

        :return:
        """

        class ImageDownloader(AshFramework):
            """协程助推器 提高挑战图片的下载效率"""

            async def control_driver(self, context, session=None):
                path_challenge_img, url = context

                # 下载挑战图片
                async with session.get(url) as response:
                    with open(path_challenge_img, "wb") as file:
                        file.write(await response.read())

        # 初始化挑战图片下载目录
        workspace_ = self._init_workspace()

        # 初始化数据容器
        docker_ = []
        for alias_, url_ in self.alias2url.items():
            path_challenge_img_ = os.path.join(workspace_, f"{alias_}.png")
            self.alias2path.update({alias_: path_challenge_img_})
            docker_.append((path_challenge_img_, url_))

        # 初始化图片下载器
        self.log(message="Download the challenge image")
        if sys.platform.startswith("win") or "cygwin" in sys.platform:
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
            asyncio.run(ImageDownloader(docker=docker_).subvert(workers="fast"))
        else:
            loop = asyncio.get_event_loop()
            loop.run_until_complete(ImageDownloader(docker=docker_).subvert(workers="fast"))

        self.runtime_workspace = workspace_

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
        # self.log(message="Start the challenge")

        # {{< IMAGE CLASSIFICATION >}}
        ta = []
        for alias, img_filepath in self.alias2path.items():
            # 读取二进制数据编织成模型可接受的类型
            with open(img_filepath, "rb") as file:
                data = file.read()

            # 获取识别结果
            t0 = time.time()
            result = model.solution(img_stream=data, label=self.label_alias[self.label])
            ta.append(time.time() - t0)

            # 模型会根据置信度给出图片中的多个目标，只要命中一个就算通过
            if result:
                # 选中标签元素
                try:
                    self.alias2locator[alias].click()
                except WebDriverException as err:
                    self.log("Failed to click on element", alias=alias, err=err)

        # {{< SUBMIT ANSWER >}}
        try:
            WebDriverWait(ctx, 35, ignored_exceptions=ElementClickInterceptedException).until(
                EC.element_to_be_clickable((By.XPATH, "//div[@class='button-submit button']"))
            ).click()
        except WebDriverException as err:
            raise ChallengeTimeout(
                "CPU computing power is insufficient "
                "to complete the challenge within the time limit"
            ) from err

        self.log(message=f"Submit the challenge - {model.flag}: {round(sum(ta), 2)}s")

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
                WebDriverWait(ctx, 1).until(
                    EC.presence_of_element_located((By.XPATH, "//div[@class='task-image']"))
                )
                return True
            except TimeoutException:
                return False

        def is_flagged_flow():
            try:
                WebDriverWait(ctx, 1, 0.1).until(
                    EC.visibility_of_element_located((By.XPATH, "//div[@class='error-text']"))
                )
                self.threat += 1
                if urllib.request.getproxies() and self.threat >1:
                    logger.warning("Your proxy IP may have been flagged.")
                return True
            except TimeoutException:
                return False

        def is_successful_at_the_demo_site():
            """//div[contains(@class,'hcaptcha-success')]"""
            try:
                ctx.switch_to.default_content()
                WebDriverWait(ctx, 1, 0.1).until(
                    EC.visibility_of_element_located(
                        (By.XPATH, "//div[contains(@class,'hcaptcha-success')]")
                    )
                )
                return True
            except TimeoutException:
                pass

        # Necessary.
        time.sleep(2)

        # Pop prompt "Please try again".
        if is_flagged_flow():
            return self.CHALLENGE_RETRY, "重置挑战"

        if is_challenge_image_clickable():
            return self.CHALLENGE_CONTINUE, "继续挑战"

        # Work only at the demo site.
        if is_successful_at_the_demo_site():
            return self.CHALLENGE_SUCCESS, "退火成功"

        # TODO > Here you need to insert a piece of business code
        #  based on your project to determine if the challenge passes
        # 可参考思路有：断言网址变更/页面跳转/DOM刷新/意外弹窗 等
        # 这些判断都是根据具体的应用场景，具体的页面元素进行编写的
        # 单独解决 hCaptcha challenge 并不困难，困难的是在业务运行时处理
        return self.CHALLENGE_SUCCESS, "退火成功"

    def anti_checkbox(self, ctx: Chrome):
        """处理复选框"""
        for _ in range(8):
            try:
                # [👻] 进入复选框
                WebDriverWait(ctx, 2, ignored_exceptions=ElementNotVisibleException).until(
                    EC.frame_to_be_available_and_switch_to_it(
                        (By.XPATH, "//iframe[contains(@title,'checkbox')]")
                    )
                )
                # [👻] 点击复选框
                WebDriverWait(ctx, 2).until(EC.element_to_be_clickable((By.ID, "checkbox"))).click()
                self.log("Handle hCaptcha checkbox")
                return True
            except TimeoutException:
                pass
            finally:
                # [👻] 回到主线剧情
                ctx.switch_to.default_content()

    def anti_hcaptcha(self, ctx: Chrome, dir_model, onnx_prefix) -> Union[bool, str]:
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
        :param dir_model:
        :param onnx_prefix:
        :return:
        """

        # [👻] 它來了！
        try:
            for index in range(3):
                # [👻] 進入挑戰框架
                self.switch_to_challenge_frame(ctx)

                # [👻] 獲取挑戰標簽
                self.get_label(ctx)

                # [👻] 編排定位器索引
                self.mark_samples(ctx)

                # [👻] 拉取挑戰圖片
                self.download_images()

                # [👻] 滤除无法处理的挑战类别
                drop = self.tactical_retreat()
                if drop in [self.CHALLENGE_BACKCALL, self.CHALLENGE_REFRESH]:
                    ctx.switch_to.default_content()
                    return drop

                # [👻] 注册解决方案
                # 根据挑战类型自动匹配不同的模型
                model = self.switch_solution(dir_model, onnx_prefix)

                # [👻] 識別|點擊|提交
                self.challenge(ctx, model=model)

                # [👻] 輪詢控制臺響應
                result, message = self.challenge_success(ctx)
                ctx.switch_to.default_content()

                self.log("Get response", desc=result)
                if result in [self.CHALLENGE_SUCCESS, self.CHALLENGE_CRASH, self.CHALLENGE_RETRY]:
                    return result
                time.sleep(1)

        except (WebDriverException,) as err:
            logger.exception(err)
            ctx.switch_to.default_content()
            return self.CHALLENGE_CRASH


class ArmorUtils:
    @staticmethod
    def fall_in_captcha_login(ctx: Chrome) -> Optional[bool]:
        """
        判断在登录时是否遇到人机挑战

        :param ctx:
        :return: True：已进入人机验证页面，False：跳转到个人主页
        """
        threshold_timeout = 35
        start = time.time()
        flag_ = ctx.current_url
        while True:
            if ctx.current_url != flag_:
                return False

            if time.time() - start > threshold_timeout:
                raise AssertTimeout("任务超时：判断是否陷入人机验证")

            try:
                ctx.switch_to.frame(
                    ctx.find_element(By.XPATH, "//iframe[contains(@title,'content')]")
                )
                ctx.find_element(By.XPATH, "//div[@class='prompt-text']")
                return True
            except WebDriverException:
                pass
            finally:
                ctx.switch_to.default_content()

    @staticmethod
    def fall_in_captcha_runtime(ctx: Chrome) -> Optional[bool]:
        """捕获隐藏在周免游戏订单中的人机挑战"""
        try:
            WebDriverWait(ctx, 5, ignored_exceptions=WebDriverException).until(
                EC.presence_of_element_located((By.XPATH, "//iframe[contains(@title,'content')]"))
            )
            return True
        except TimeoutException:
            return False

    @staticmethod
    def face_the_checkbox(ctx: Chrome) -> Optional[bool]:
        """遇见 hCaptcha checkbox"""
        try:
            WebDriverWait(ctx, 8, ignored_exceptions=WebDriverException).until(
                EC.presence_of_element_located((By.XPATH, "//iframe[contains(@title,'checkbox')]"))
            )
            return True
        except TimeoutException:
            return False
