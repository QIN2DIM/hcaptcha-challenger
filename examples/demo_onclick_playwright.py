# -*- coding: utf-8 -*-
# Time       : 2023/8/20 23:12
# Author     : QIN2DIM
# GitHub     : https://github.com/QIN2DIM
# Description:
from __future__ import annotations

import json
import time
from contextlib import suppress
from pathlib import Path

from loguru import logger
from playwright.sync_api import BrowserContext, TimeoutError

import hcaptcha_challenger as solver
from hcaptcha_challenger.agents.exceptions import ChallengePassed
from hcaptcha_challenger.agents.playwright import Tarnished
from hcaptcha_challenger.agents.playwright.onclick_challenge import OnClickAgent

# Init local-side of the ModelHub
solver.install()

# Save dataset to current working directory
tmp_dir = Path(__file__).parent.joinpath("tmp_dir")
user_data_dir = Path(__file__).parent.joinpath("user_data_dir")
context_dir = user_data_dir.joinpath("context")
record_dir = user_data_dir.joinpath("record")
record_har_path = record_dir.joinpath(f"eg-{int(time.time())}.har")


class SiteKey:
    discord = "f5561ba9-8f1e-40ca-9b5b-a0b3f719ef34"
    epic = "91e4137f-95af-4bc9-97af-cdcedce21c8c"
    hcaptcha = "a5f74b19-9e45-40e0-b45d-47ff91b7a6c2"
    hcaptcha_signup = "13257c82-e129-4f09-a733-2a7cb3102832"
    new_type_challenge = "ace50dd0-0d68-44ff-931a-63b670c7eed7"
    user = "c86d730b-300a-444c-a8c5-5312e7a93628"
    cloud_horse = "edc4ce89-8903-4906-80b1-7440ad9a69c8"
    top_level = "adafb813-8b5c-473f-9de3-485b4ad5aa09"

    @staticmethod
    def to_sitelink() -> str:
        return f"https://accounts.hcaptcha.com/demo?sitekey={SiteKey.new_type_challenge}"


@logger.catch
def hit_challenge(context: BrowserContext):
    agent = OnClickAgent.from_modelhub(tmp_dir=tmp_dir)

    page = context.pages[0]
    agent.handle_onclick_resp(page)
    page.goto(SiteKey.to_sitelink())

    with suppress(TimeoutError):
        page.locator("//iframe[contains(@title,'checkbox')]").wait_for()
        agent.anti_checkbox(page)

    for _ in range(8):
        with suppress(ChallengePassed):
            result = agent.anti_hcaptcha(page)
            print(f">> Challenge Result: {result}")
            if result == agent.status.CHALLENGE_BACKCALL:
                fl = page.frame_locator(agent.HOOK_CHALLENGE)
                fl.locator("//div[@class='refresh button']").click()
                continue
            if result == agent.status.CHALLENGE_SUCCESS:
                rqdata_path = Path("tmp_dir", f"rqdata-{time.time()}.json")
                rqdata_path.write_text(json.dumps(agent.challenge_resp.__dict__, indent=2))
                print(f"View RQdata path={rqdata_path}")
                page.wait_for_timeout(2000)
                return


def bytedance():
    radagon = Tarnished(
        user_data_dir=context_dir, record_dir=record_dir, record_har_path=record_har_path
    )
    radagon.execute(sequence=[hit_challenge])
    print(f"View record video path={record_dir}")


if __name__ == "__main__":
    bytedance()
