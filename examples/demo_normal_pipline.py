# -*- coding: utf-8 -*-
# Time       : 2023/11/21 6:06
# Author     : QIN2DIM
# GitHub     : https://github.com/QIN2DIM
# Description:
from __future__ import annotations

import asyncio
import json
from pathlib import Path

from hcaptcha_challenger import install, QuestionResp, Answers, Status
from hcaptcha_challenger.agents import AgentR
from loguru import logger

WARN = """
The demo here uses a static file and it cannot run

Because the public network access permission for each challenge-image has expired,
the images cannot be downloaded.

Get real-time challenge tasks from https://api.hcaptcha.com/getcaptcha/｛sitekey｝
you should know how to do it

"""


def prelude() -> AgentR:
    # You need to deploy sub-thread tasks and
    # actively run `install(upgrade=True)` every 20 minutes
    install(upgrade=True, clip=True)

    # You need to make sure to run `install(upgrade=True, clip=True)`
    # before each instantiation
    agent = AgentR.summon_ranni_the_witch(
        # Mount the cache directory to the current working folder
        tmp_dir=Path(__file__).parent.joinpath("tmp_dir"),
        # Enable CLIP zero-shot image classification method
        clip=True,
    )

    return agent


def get_question_data() -> QuestionResp:
    """

    ```python
    import httpx


    url = "https://api.hcaptcha.com/getcaptcha/｛sitekey｝"
    res = httpx.post(url, json=payload)
    data = res.json()
    qr = QuestionResp(**data)

    ```

    :return:
    """
    dp = Path(__file__).parent.parent.joinpath(
        "assets/record_json/image_label_binary.beverage.json"
    )
    data = json.loads(dp.read_text(encoding="utf8"))
    return QuestionResp(**data)


async def main():
    # Obtain challenge tasks in a way you are familiar with
    qr = get_question_data()

    # An agent instance can run multiple execute tasks,
    # so you don't have to instantiate AgentR frequently.
    agent = prelude()

    # Run multimodal tasks
    response: Answers | Status | None = await agent.execute(qr)

    if isinstance(response, agent.status):
        logger.success(f"task done", response=response)
    elif response and hasattr(response, "answers"):
        logger.warning(WARN, answers=response.answers)
    else:
        logger.error("response is None", response=response)


if __name__ == "__main__":
    asyncio.run(main())
