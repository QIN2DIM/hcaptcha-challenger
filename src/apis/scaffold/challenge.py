# -*- coding: utf-8 -*-
# Time       : 2022/1/16 0:25
# Author     : QIN2DIM
# Github     : https://github.com/QIN2DIM
# Description:
import time
import warnings
from typing import Optional

from selenium.common.exceptions import WebDriverException

from services.hcaptcha_challenger import ArmorCaptcha, ArmorUtils
from services.hcaptcha_challenger.exceptions import ChallengePassed
from services.settings import (
    logger,
    HCAPTCHA_DEMO_SITES,
    DIR_MODEL,
    DIR_CHALLENGE,
    PATH_OBJECTS_YAML,
)
from services.utils import get_challenge_ctx

warnings.filterwarnings("ignore", category=DeprecationWarning)


@logger.catch()
def runner(
    sample_site: str,
    lang: Optional[str] = "zh",
    silence: Optional[bool] = False,
    onnx_prefix: Optional[str] = None,
    screenshot: Optional[bool] = False,
):
    """Human-Machine Challenge Demonstration | Top Interface"""

    # Instantiating Challenger Components
    challenger = ArmorCaptcha(
        dir_workspace=DIR_CHALLENGE,
        lang=lang,
        debug=True,
        dir_model=DIR_MODEL,
        onnx_prefix=onnx_prefix,
        screenshot=screenshot,
        path_objects_yaml=PATH_OBJECTS_YAML,
        on_rainbow=True,
    )
    challenger_utils = ArmorUtils()

    # Instantiating the Challenger Drive
    ctx = get_challenge_ctx(silence=silence, lang=lang)
    _round = 5
    try:
        for i in range(_round):
            try:
                # Read the hCaptcha challenge test site
                ctx.get(sample_site)

                # Detects if a clickable `hcaptcha checkbox` appears on the current page.
                # The `sample site` must pop up the `checkbox`, where the flexible wait time defaults to 5s.
                # If the `checkbox` does not load in 5s, your network is in a bad state.
                if not challenger_utils.face_the_checkbox(ctx):
                    break

                start = time.time()

                # Enter iframe-checkbox --> Process hcaptcha checkbox --> Exit iframe-checkbox
                challenger.anti_checkbox(ctx)

                # Enter iframe-content --> process hcaptcha challenge --> exit iframe-content
                resp = challenger.anti_hcaptcha(ctx)
                if resp == challenger.CHALLENGE_SUCCESS:
                    challenger.log(f"End of demo - total: {round(time.time() - start, 2)}s")
                    logger.success(f"PASS[{i + 1}|{_round}]".center(28, "="))
                elif resp == challenger.CHALLENGE_RETRY:
                    ctx.refresh()
                    logger.error(f"RETRY[{i + 1}|{_round}]".center(28, "="))

            except ChallengePassed:
                ctx.refresh()
                logger.success(f"PASS[{i + 1}|{_round}]".center(28, "="))
            except WebDriverException as err:
                logger.exception(err)
    finally:
        ctx.quit()


@logger.catch()
def test():
    """Check if the Challenger driver version is compatible"""
    ctx = get_challenge_ctx(silence=True)
    try:
        ctx.get(HCAPTCHA_DEMO_SITES[0])
    finally:
        ctx.quit()

    logger.success("The adaptation is successful")
