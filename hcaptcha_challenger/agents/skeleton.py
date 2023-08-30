# -*- coding: utf-8 -*-
# Time       : 2023/8/20 0:16
# Author     : QIN2DIM
# GitHub     : https://github.com/QIN2DIM
# Description:
from __future__ import annotations

import time
from abc import ABC
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Literal
from urllib.parse import quote

from loguru import logger

from hcaptcha_challenger.onnx.modelhub import ModelHub
from hcaptcha_challenger.onnx.resnet import ResNetControl
from hcaptcha_challenger.onnx.yolo import YOLOv8, apply_ash_of_war

HOOK_CHALLENGE = "//iframe[contains(@src,'#frame=challenge')]"


@dataclass
class Status:
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
    # <to-X> NOT MATCH PATTERN
    CHALLENGE_TO_BINARY = "to_binary"
    CHALLENGE_TO_AREA_SELECT = "to_area_select"

    AUTH_SUCCESS = "success"
    AUTH_ERROR = "error"
    AUTH_CHALLENGE = "challenge"


@dataclass
class Skeleton(ABC):
    """hCAPTCHA challenge drive control"""

    modelhub: ModelHub
    """
    Build Skeleton with modelhub
    """

    this_dir: Path = Path(__file__).parent
    """
    Project directory of Skeleton Agents
    """

    tmp_dir: Path = this_dir.joinpath("temp_cache")
    challenge_dir = tmp_dir.joinpath("_challenge")
    """
    Runtime cache
    """

    _label = ""
    """
    Cleaned Challenge Prompt in the context
    """

    _prompt = ""
    """
    Challenge Prompt in the context
    """

    _label_alias: Dict[str, str] = field(default_factory=dict)
    """
    A collection of { prompt[s]: model_name[.onnx] }
    """

    _alias2locator: dict = field(default_factory=dict)
    """
    Store the `element locator` of challenge images {挑战图片1: locator1, ...}
    """

    _alias2url: dict = field(default_factory=dict)
    """
    Store the `download link` of the challenge image {挑战图片1: url1, ...}
    """

    _alias2path: dict = field(default_factory=dict)
    """
    Store the `directory` of challenge image {挑战图片1: "/images/挑战图片1.png", ...}
    """

    @property
    def status(self):
        return Status

    @classmethod
    def from_modelhub(cls, tmp_dir: Path | None = None, **kwargs):
        modelhub = ModelHub.from_github_repo(**kwargs)
        modelhub.parse_objects()

        dragon = cls(modelhub=modelhub, _label_alias=modelhub.label_alias)

        if tmp_dir and isinstance(tmp_dir, Path):
            dragon.tmp_dir = tmp_dir
            dragon.challenge_dir = tmp_dir.joinpath("_challenge")

        return dragon

    def _match_solution(self, select: Literal["yolo", "resnet"] = None) -> ResNetControl | YOLOv8:
        """match solution after `tactical_retreat`"""
        focus_label = self._label_alias.get(self._label, "")

        # Match YOLOv8 model
        if not focus_label or select == "yolo":
            focus_name = apply_ash_of_war(ash=self._label)
            session = self.modelhub.match_net(focus_name=focus_name)
            detector = YOLOv8.from_pluggable_model(session, focus_name)
            return detector

        # Match ResNet model
        focus_name = focus_label
        if not focus_name.endswith(".onnx"):
            focus_name = f"{focus_name}.onnx"
        net = self.modelhub.match_net(focus_name=focus_name)
        control = ResNetControl.from_pluggable_model(net)
        return control

    def switch_to_challenge_frame(self, ctx, **kwargs):
        raise NotImplementedError

    def get_label(self, ctx, **kwargs):
        """Obtain the label that needs to be recognized for the challenge"""
        raise NotImplementedError

    def tactical_retreat(self, **kwargs) -> str | None:
        """skip unchoreographed challenges"""
        if self._label_alias.get(self._label):
            return self.status.CHALLENGE_CONTINUE

        q = quote(self._label, "utf8")
        logger.warning(
            "Types of challenges not yet scheduled",
            label=self._label,
            prompt=self._prompt,
            issue=f"https://github.com/QIN2DIM/hcaptcha-challenger/issues?q={q}",
        )
        return self.status.CHALLENGE_BACKCALL

    def mark_samples(self, ctx, *args, **kwargs):
        """Get the download link and locator of each challenge image"""
        raise NotImplementedError

    def download_images(self):
        prefix = ""
        if self._label:
            prefix = f"{time.time()}_{self._label_alias.get(self._label, self._label)}"
        runtime_dir = self.challenge_dir.joinpath(prefix)
        runtime_dir.mkdir(mode=777, parents=True, exist_ok=True)

        # Initialize the data container
        container = []
        for alias_, url_ in self._alias2url.items():
            challenge_img_path = runtime_dir.joinpath(f"{alias_}.png")
            self._alias2path.update({alias_: challenge_img_path})
            container.append((challenge_img_path, url_))

        return container

    def challenge(self, ctx, model, *args, **kwargs):
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
        raise NotImplementedError

    def is_success(self, ctx, *args, **kwargs):
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
        raise NotImplementedError

    def anti_checkbox(self, ctx, *args, **kwargs):
        raise NotImplementedError

    def anti_hcaptcha(self, ctx, *args, **kwargs) -> bool | str:
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
        raise NotImplementedError
