# -*- coding: utf-8 -*-
# Time       : 2022/1/16 0:25
# Author     : QIN2DIM
# Github     : https://github.com/QIN2DIM
# Description:
import time
from typing import Optional

from services.hcaptcha_challenger import ArmorCaptcha, ArmorUtils, YOLO
from services.settings import logger, HCAPTCHA_DEMO_SITES, DIR_MODEL, DIR_CHALLENGE
from services.utils import get_challenge_ctx


def demo(
    silence: Optional[bool] = False,
    onnx_prefix: Optional[str] = None,
    sample_site: Optional[str] = HCAPTCHA_DEMO_SITES[0],
    wait_: Optional[bool] = False,
):
    """人机挑战演示 顶级接口"""
    logger.info("Starting demo project...")

    # 实例化嵌入式模型
    yolo = YOLO(DIR_MODEL, onnx_prefix=onnx_prefix)

    # 实例化挑战者组件
    challenger = ArmorCaptcha(dir_workspace=DIR_CHALLENGE, debug=True)
    challenger_utils = ArmorUtils()

    # 实例化挑战者驱动
    ctx = get_challenge_ctx(silence=silence)
    try:
        # 读取 hCaptcha challenge 测试站点
        ctx.get(sample_site)

        # 必要的等待时间
        time.sleep(3)

        if wait_:
            input(
                "Please manually clear all masking elements to ensure that "
                "the man-machine challenge is fully displayed before continuing to run."
            )

        # 检测当前页面是否出现可点击的 `hcaptcha checkbox`
        # `样本站点` 必然会弹出 `checkbox`，此处的弹性等待时长默认为 5s，
        # 若 5s 仍未加载出 `checkbox` 说明您当前的网络状态堪忧
        if challenger_utils.face_the_checkbox(ctx):
            start = time.time()

            # 进入 iframe-checkbox --> 处理 hcaptcha checkbox --> 退出 iframe-checkbox
            challenger.anti_checkbox(ctx)

            # 进入 iframe-content --> 处理 hcaptcha challenge --> 退出 iframe-content
            challenger.anti_hcaptcha(ctx, model=yolo)

            challenger.log(f"演示结束，挑战总耗时：{round(time.time() - start, 2)}s")
    finally:
        input("[EXIT] Press any key to exit...")

        ctx.quit()


def demo_v2(silence: Optional[bool] = False, onnx_prefix: Optional[str] = None):
    """
    人机挑战演示 顶级接口 演示垂直河流 SKI 解法

    :param silence:
    :param onnx_prefix:
    :return:
    """
    demo(silence, onnx_prefix, sample_site=HCAPTCHA_DEMO_SITES[1], wait_=True)


def demo_v3(silence: Optional[bool] = False, onnx_prefix: Optional[str] = None):
    """
    人机挑战演示 顶级接口 演示天空中向左飞行的飞机的 SKI 解法

    :param silence:
    :param onnx_prefix:
    :return:
    """
    demo(silence, onnx_prefix, sample_site=HCAPTCHA_DEMO_SITES[2], wait_=True)


@logger.catch()
def test():
    """检查挑战者驱动版本是否适配"""
    ctx = get_challenge_ctx(silence=True)
    try:
        ctx.get(HCAPTCHA_DEMO_SITES[0])
    finally:
        ctx.quit()

    logger.success("The adaptation is successful")
