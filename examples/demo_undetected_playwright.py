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
from playwright.async_api import BrowserContext as ASyncContext, async_playwright

import hcaptcha_challenger as solver
from hcaptcha_challenger.agents import AgentT, Malenia
from hcaptcha_challenger.utils import SiteKey

# Init local-side of the ModelHub
solver.install(upgrade=True)

# Save dataset to current working directory
tmp_dir = Path(__file__).parent.joinpath("tmp_dir")
user_data_dir = Path(__file__).parent.joinpath("user_data_dir")
context_dir = user_data_dir.joinpath("context")
record_video_dir = user_data_dir.joinpath("record")
record_har_path = record_video_dir.joinpath(f"eg-{int(time.time())}.har")

sitekey = SiteKey.user_easy


@logger.catch
async def hit_challenge(context: ASyncContext, times: int = 8):
    page = context.pages[0]
    agent = AgentT.from_page(page=page, tmp_dir=tmp_dir)
    await page.goto(SiteKey.as_sitelink(sitekey))

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
    # playwright install firefox --with-deps
    async with async_playwright() as p:
        context = await p.firefox.launch_persistent_context(
            user_data_dir=context_dir,
            headless=False,
            locale="en-US",
            record_video_dir=record_video_dir,
            record_har_path=record_har_path,
        )
        await Malenia.apply_stealth(context)
        await hit_challenge(context)
        print(f"View record video path={record_video_dir}")


if __name__ == "__main__":
    asyncio.run(bytedance())
