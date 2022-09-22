# -*- coding: utf-8 -*-
# Time       : 2022/1/16 0:25
# Author     : QIN2DIM
# Github     : https://github.com/QIN2DIM
# Description:
import time
import typing
import warnings

from loguru import logger
from selenium.common.exceptions import WebDriverException

from services.hcaptcha_challenger import HolyChallenger
from services.hcaptcha_challenger.exceptions import ChallengePassed
from services.settings import HCAPTCHA_DEMO_SITES, DIR_MODEL, DIR_CHALLENGE, PATH_OBJECTS_YAML
from services.utils import get_challenge_ctx

warnings.filterwarnings("ignore", category=DeprecationWarning)


@logger.catch()
def test():
    """Check if the Challenger driver version is compatible"""
    ctx = get_challenge_ctx(silence=True)
    try:
        ctx.get(HCAPTCHA_DEMO_SITES[0])
    finally:
        ctx.quit()
    logger.success("The adaptation is successful")


def _motion(sample_site: str, ctx, challenger: HolyChallenger) -> typing.Optional[str]:
    resp = None

    # Read the hCaptcha challenge test site
    ctx.get(sample_site)

    # Detects if a clickable `hcaptcha checkbox` appears on the current page.
    # The `sample site` must pop up the `checkbox`, where the flexible wait time defaults to 5s.
    # If the `checkbox` does not load in 5s, your network is in a bad state.
    if challenger.utils.face_the_checkbox(ctx):
        # Enter iframe-checkbox --> Process hcaptcha checkbox --> Exit iframe-checkbox
        challenger.anti_checkbox(ctx)
        # Enter iframe-content --> process hcaptcha challenge --> exit iframe-content
        resp = challenger.anti_hcaptcha(ctx)
    return resp


@logger.catch()
def runner(
    sample_site: str,
    lang: typing.Optional[str] = "zh",
    silence: typing.Optional[bool] = False,
    onnx_prefix: typing.Optional[str] = None,
    screenshot: typing.Optional[bool] = False,
    repeat: typing.Optional[int] = 10,
):
    """Human-Machine Challenge Demonstration | Top Interface"""

    # Instantiating Challenger Components
    challenger = HolyChallenger(
        dir_workspace=DIR_CHALLENGE,
        dir_model=DIR_MODEL,
        path_objects_yaml=PATH_OBJECTS_YAML,
        onnx_prefix=onnx_prefix,
        screenshot=screenshot,
        lang=lang,
        debug=True,
    )

    # Instantiating the Challenger Drive
    with get_challenge_ctx(silence=silence, lang=lang) as ctx:
        for i in range(repeat):
            start = time.time()
            try:
                if (resp := _motion(sample_site, ctx=ctx, challenger=challenger)) is None:
                    logger.warning("UnknownMistake")
                elif resp == challenger.CHALLENGE_SUCCESS:
                    challenger.log(f"End of demo - total: {round(time.time() - start, 2)}s")
                    logger.success(f"PASS[{i + 1}|{repeat}]".center(28, "="))
                elif resp == challenger.CHALLENGE_RETRY:
                    ctx.refresh()
                    logger.error(f"RETRY[{i + 1}|{repeat}]".center(28, "="))
            except ChallengePassed:
                ctx.refresh()
                logger.success(f"PASS[{i + 1}|{repeat}]".center(28, "="))
            except WebDriverException as err:
                logger.warning(err)
