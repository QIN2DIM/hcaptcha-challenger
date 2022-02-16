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


def demo(silence: Optional[bool] = False, onnx_prefix="yolov5s6"):
    """人机挑战演示 顶级接口"""
    logger.info("Starting demo project...")

    # 指定嵌入式模型
    yolo = YOLO(DIR_MODEL, onnx_prefix=onnx_prefix)

    # 实例化挑战者组件
    challenger = ArmorCaptcha(dir_workspace=DIR_CHALLENGE, debug=True)
    challenger_utils = ArmorUtils()

    # 实例化挑战者驱动
    ctx = get_challenge_ctx(silence=silence)
    try:
        # 读取 hCaptcha challenge 测试站点
        # 默认使用 maximedrn 提供的测试站点
        demo_url = HCAPTCHA_DEMO_SITES[0]
        ctx.get(demo_url)

        time.sleep(3)

        # 检测当前页面是否出现可点击的 hcaptcha checkbox
        # 测试站点必然会弹出 checkbox，此处的弹性等待时长默认为 5s，
        # 若 5s 仍未加载出 checkbox 说明您当前的网络状态堪忧
        if challenger_utils.face_the_checkbox(ctx):
            start = time.time()

            # 进入 iframe-checkbox --> 处理 hcaptcha checkbox --> 退出 iframe-checkbox
            challenger.anti_checkbox(ctx)

            # 进入 iframe-content --> 处理 hcaptcha challenge --> 退出 iframe-content
            challenger.anti_hcaptcha(ctx, model=yolo)

            challenger.log(f"演示结束，挑战总耗时：{round(time.time() - start, 2)}s")
    finally:
        input(f"[EXIT] Press any key to exit...")

        ctx.quit()


@logger.catch()
def test():
    """检查挑战者驱动版本是否适配"""
    ctx = get_challenge_ctx(silence=True)
    try:
        ctx.get("https://www.baidu.com")
    finally:
        ctx.quit()

    logger.success("The adaptation is successful")


if __name__ == '__main__':
    demo()
