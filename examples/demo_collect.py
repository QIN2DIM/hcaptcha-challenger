# -*- coding: utf-8 -*-
# Time       : 2023/8/31 20:54
# Author     : QIN2DIM
# GitHub     : https://github.com/QIN2DIM
# Description:
import asyncio
from pathlib import Path

from loguru import logger
from playwright.async_api import BrowserContext as ASyncContext, async_playwright

from hcaptcha_challenger.agents.playwright.control import CaptchaAgent
from hcaptcha_challenger.agents.playwright.stealthened import Stealthened
from hcaptcha_challenger.utils import SiteKey

# Save dataset to current working directory
tmp_dir = Path(__file__).parent.joinpath("tmp_dir")
user_data_dir = Path(__file__).parent.joinpath("user_data_dir")
context_dir = user_data_dir.joinpath("context")

labels = set()


@logger.catch
async def collete_datasets(context: ASyncContext, batch: int = 80):
    page = await context.new_page()
    agent = CaptchaAgent.from_page(page=page, tmp_dir=tmp_dir)

    await page.goto(SiteKey.as_sitelink(sitekey="adafb813-8b5c-473f-9de3-485b4ad5aa09"))

    await agent.handle_checkbox()

    for pth in range(1, batch + 1):
        label = await agent.collect()
        labels.add(label)
        print(f"\r>> COLLETE - progress={pth}/{batch} {label=}", end="")
        await page.wait_for_timeout(500)
        fl = page.frame_locator(agent.HOOK_CHALLENGE)
        await fl.locator("//div[@class='refresh button']").click()


async def bytedance():
    malenia = Stealthened(user_data_dir=context_dir)
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(locale="en-US")
        await malenia.apply_stealth(context)
        await collete_datasets(context)
        await context.close()

    print(f"\n>> COUNT - {labels=}")


if __name__ == "__main__":
    asyncio.run(bytedance())
