# -*- coding: utf-8 -*-
# Time       : 2022/1/16 0:25
# Author     : QIN2DIM
# Github     : https://github.com/QIN2DIM
# Description:
import time
from typing import Optional

from selenium.common.exceptions import WebDriverException

from services.hcaptcha_challenger import ArmorCaptcha, ArmorUtils, YOLO
from services.hcaptcha_challenger.exceptions import ChallengeReset
from services.settings import logger, HCAPTCHA_DEMO_SITES, DIR_MODEL, DIR_CHALLENGE
from services.utils import get_challenge_ctx


@logger.catch()
def runner(
    sample_site: str,
    lang: Optional[str] = "zh",
    silence: Optional[bool] = False,
    onnx_prefix: Optional[str] = None,
):
    """人机挑战演示 顶级接口"""
    logger.info("Starting demo project...")

    # Instantiating embedded models
    yolo = YOLO(DIR_MODEL, onnx_prefix=onnx_prefix)

    # Instantiating Challenger Components
    challenger = ArmorCaptcha(dir_workspace=DIR_CHALLENGE, lang=lang, debug=True)
    challenger_utils = ArmorUtils()

    # Instantiating the Challenger Drive
    ctx = get_challenge_ctx(silence=silence, lang=lang)
    try:
        for _ in range(5):
            try:
                # Read the hCaptcha challenge test site
                ctx.get(sample_site)

                # Necessary waiting time
                time.sleep(3)

                # Detects if a clickable `hcaptcha checkbox` appears on the current page.
                # The `sample site` must pop up the `checkbox`, where the flexible wait time defaults to 5s.
                # If the `checkbox` does not load in 5s, your network is in a bad state.
                if challenger_utils.face_the_checkbox(ctx):
                    start = time.time()

                    # Enter iframe-checkbox --> Process hcaptcha checkbox --> Exit iframe-checkbox
                    challenger.anti_checkbox(ctx)

                    # Enter iframe-content --> process hcaptcha challenge --> exit iframe-content
                    result = challenger.anti_hcaptcha(ctx, model=yolo)
                    if not result:
                        continue

                    challenger.log(
                        f"End of demo - total: {round(time.time() - start, 2)}s"
                    )

                break

            # Do not capture the `ChallengeReset` signal in the outermost layer.
            # In the demo project, we wanted the human challenge to pop up, not pass after processing the checkbox.
            # So when this happens, we reload the page to activate hcaptcha repeatedly.
            # But in your project, if you've passed the challenge by just handling the checkbox,
            # there's no need to refresh the page!
            except (WebDriverException, ChallengeReset):
                continue
    finally:
        input("[EXIT] Press any key to exit...")

        ctx.quit()


@logger.catch()
def test():
    """检查挑战者驱动版本是否适配"""
    ctx = get_challenge_ctx(silence=True)
    try:
        ctx.get(HCAPTCHA_DEMO_SITES[0])
    finally:
        ctx.quit()

    logger.success("The adaptation is successful")
