# -*- coding: utf-8 -*-
# Time       : 2023/8/31 20:54
# Author     : QIN2DIM
# GitHub     : https://github.com/QIN2DIM
# Description:
import asyncio
import time
from collections import Counter
from contextlib import suppress
from pathlib import Path

from loguru import logger
from playwright.async_api import BrowserContext as ASyncContext, async_playwright

from hcaptcha_challenger import Malenia, AgentT
from hcaptcha_challenger.utils import SiteKey

collected = []
per_times = 80
tmp_dir = Path(__file__).parent.joinpath("tmp_dir")
sitekey = SiteKey.user_easy


async def collete_datasets(context: ASyncContext):
    page = await context.new_page()
    agent = AgentT.from_page(page=page, tmp_dir=tmp_dir)

    sitelink = SiteKey.as_sitelink(sitekey)
    await page.goto(sitelink)

    await agent.handle_checkbox()

    for pth in range(1, per_times + 1):
        with suppress(Exception):
            t0 = time.time()
            label = await agent.collect()
            te = f"{time.time() - t0:.2f}s"
            probe = list(agent.qr.requester_restricted_answer_set.keys())
            mixed_label = probe[0] if len(probe) > 0 else label
            collected.append(mixed_label)
            print(f">> COLLETE - progress=[{pth}/{per_times}] timeit={te} {label=} {probe=}")

        await page.wait_for_timeout(500)
        fl = page.frame_locator(agent.HOOK_CHALLENGE)
        await fl.locator("//div[@class='refresh button']").click()


@logger.catch
async def bytedance():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(locale="en-US")
        await Malenia.apply_stealth(context)
        await collete_datasets(context)
        await context.close()

    print(f"\n>> RESULT - {Counter(collected)=}")


if __name__ == "__main__":
    asyncio.run(bytedance())
