# -*- coding: utf-8 -*-
# Time       : 2023/8/20 23:12
# Author     : QIN2DIM
# GitHub     : https://github.com/QIN2DIM
# Description:
from __future__ import annotations

import asyncio
import time
from pathlib import Path

from loguru import logger
from playwright.async_api import BrowserContext as ASyncContext

import hcaptcha_challenger as solver
from hcaptcha_challenger.agents.playwright.control import AgentT
from hcaptcha_challenger.agents.playwright.tarnished import Malenia

# Init local-side of the ModelHub
solver.install(flush_yolo=True)

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
        return f"https://accounts.hcaptcha.com/demo?sitekey={SiteKey.user}"


@logger.catch
async def hit_challenge(context: ASyncContext, times: int = 8):
    page = context.pages[0]
    agent = AgentT.from_page(page=page, tmp_dir=tmp_dir)
    await page.goto(SiteKey.to_sitelink())

    await agent.handle_checkbox()

    for pth in range(1, times):
        result = await agent()
        print(f">> {pth} - Challenge Result: {result}")
        match result:
            case agent.status.CHALLENGE_BACKCALL:
                await page.wait_for_timeout(500)
                fl = page.frame_locator(agent.HOOK_CHALLENGE)
                await fl.locator("//div[@class='refresh button']").click()
            case agent.status.CHALLENGE_SUCCESS:
                rqdata_path = agent.export_rq()
                print(f"View RQdata path={rqdata_path}")
                return


async def bytedance():
    malenia = Malenia(
        user_data_dir=context_dir, record_dir=record_dir, record_har_path=record_har_path
    )
    await malenia.execute(sequence=[hit_challenge], headless=False)
    print(f"View record video path={record_dir}")


if __name__ == "__main__":
    asyncio.run(bytedance())
