# -*- coding: utf-8 -*-
# Time       : 2022/1/16 0:25
# Author     : QIN2DIM
# Github     : https://github.com/QIN2DIM
# Description:
import time
import typing

from loguru import logger

import hcaptcha_challenger as solver
from hcaptcha_challenger.exceptions import ChallengePassed


@logger.catch()
def test(hcaptcha_demo_site: str = "https://accounts.hcaptcha.com/demo"):
    """Check if the Challenger driver version is compatible"""
    ctx = solver.get_challenge_ctx(silence=True)
    try:
        ctx.get(hcaptcha_demo_site)
    finally:
        ctx.quit()
    logger.success("The adaptation is successful")


def _motion(sample_site: str, ctx, challenger: solver.HolyChallenger) -> typing.Optional[str]:
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
def run(
    sample_site: str,
    lang: typing.Optional[str] = "zh",
    silence: typing.Optional[bool] = False,
    onnx_prefix: typing.Optional[str] = None,
    screenshot: typing.Optional[bool] = False,
    repeat: typing.Optional[int] = 10,
    slowdown: typing.Optional[bool] = True,
):
    """Human-Machine Challenge Demonstration | Top Interface"""

    # Instantiating Challenger Components
    challenger = solver.new_challenger(
        screenshot=screenshot,
        debug=True,
        lang=lang,
        onnx_prefix=onnx_prefix or "yolov6n",
        slowdown=slowdown,
    )
    ctx = solver.get_challenge_ctx(silence=silence, lang=lang)
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


if __name__ == "__main__":
    run()
