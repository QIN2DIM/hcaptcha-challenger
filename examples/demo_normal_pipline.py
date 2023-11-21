# -*- coding: utf-8 -*-
# Time       : 2023/11/21 6:06
# Author     : QIN2DIM
# GitHub     : https://github.com/QIN2DIM
# Description:
from __future__ import annotations

import asyncio
import json
from pathlib import Path

from loguru import logger

from hcaptcha_challenger import install, QuestionResp, Answers, Status, ModelHub
from hcaptcha_challenger.agents import AgentR

WARN = """
The demo here uses a static file and it cannot run

Because the public network access permission for each challenge-image has expired,
the images cannot be downloaded.

Get real-time challenge tasks from https://api.hcaptcha.com/getcaptcha/｛sitekey｝
you should know how to do it

`Answers` BaseModel has implemented some structure fields, and you can use it to complete the remaining fields.
You can export it as json format by the `model_dump_json()` method

"""


def patch_modelhub(modelhub: ModelHub):
    """
    1. Patching clip_candidates allows you to handle all image classification tasks in self-supervised mode.

    2. You need to inject hints for all categories that appear in a batch of images

    3. The ObjectsYaml in the GitHub repository are updated regularly,
    but if you find something new, you can imitate the following and patch some hints.

    4. Note that this should be a regularly changing table.
    If after a while certain labels no longer appear, you should not fill them in clip_candidates

    5. Please note that you only need a moderate number of candidates prompts,
    too many prompts will increase the computational complexity
    :param modelhub:
    :return:
    """

    modelhub.clip_candidates.update(
        {
            "the largest animal in real life": [
                "parrot",
                "bee",
                "ladybug",
                "frog",
                "crab",
                "bat",
                "butterfly",
                "dragonfly",
            ]
        }
    )


def prelude() -> AgentR:
    # 1. You need to deploy sub-thread tasks and actively run `install(upgrade=True)` every 20 minutes
    # 2. You need to make sure to run `install(upgrade=True, clip=True)` before each instantiation
    install(upgrade=True, clip=True)

    modelhub = ModelHub.from_github_repo()
    modelhub.parse_objects()

    # Make arbitrary pre-modifications to modelhub, which is very useful for CLIP models
    patch_modelhub(modelhub)

    agent = AgentR.summon_ranni_the_witch(
        # Register modelhub externally, and the agent can patch custom configurations
        modelhub=modelhub,
        # Mount the cache directory to the current working folder
        tmp_dir=Path(__file__).parent.joinpath("tmp_dir"),
        # Enable CLIP zero-shot image classification method
        clip=True,
    )

    return agent


def get_question_data() -> QuestionResp:
    """
    import httpx

    url = "https://api.hcaptcha.com/getcaptcha/｛sitekey｝"
    res = httpx.post(url, json=payload)
    data = res.json()

    :return: QuestionResp(**data)
    """
    dp = Path(__file__).parent.parent.joinpath(
        "assets/record_json/image_label_binary.beverage.json"
    )
    data = json.loads(dp.read_text(encoding="utf8"))
    return QuestionResp(**data)


def cache_rqdata(cache_dir: Path, response: Answers):
    record_json = cache_dir.joinpath("payload_json")
    record_json.mkdir(parents=True, exist_ok=True)

    # You can export it as json format by the `model_dump_json()` method
    json_path = record_json.joinpath(f"pipline-{response.job_mode}.json")
    json_path.write_text(response.model_dump_json(indent=2))

    logger.success("cache rqdata", output=f"{json_path}")


async def main(qr: QuestionResp | None = None):
    # Obtain real-time challenge tasks in a way you are familiar with
    qr = qr or get_question_data()

    # An agent instance can run multiple execute tasks,
    # so you don't have to instantiate AgentR frequently.
    agent = prelude()

    # Run multimodal tasks
    response: Answers | Status | None = await agent.execute(qr)

    # Handle response
    if isinstance(response, agent.status):
        logger.success("task done", status=response)
    elif response and isinstance(response, Answers):
        logger.warning(WARN, answers=response.answers)
        cache_rqdata(agent.tmp_dir, response)
    else:
        logger.error("response is None", response=response)


if __name__ == "__main__":
    asyncio.run(main())
