# -*- coding: utf-8 -*-
# Time       : 2023/8/31 20:54
# Author     : QIN2DIM
# GitHub     : https://github.com/QIN2DIM
# Description:
import asyncio
from pathlib import Path

from playwright.async_api import BrowserContext as ASyncContext

from hcaptcha_challenger.agents.playwright.control import AgentT
from hcaptcha_challenger.agents.playwright.tarnished import Malenia
from hcaptcha_challenger.utils import SiteKey
from loguru import logger

# Save dataset to current working directory
tmp_dir = Path(__file__).parent.joinpath("tmp_dir")
user_data_dir = Path(__file__).parent.joinpath("user_data_dir")
context_dir = user_data_dir.joinpath("context")

labels = set()


@logger.catch
async def collete_datasets(context: ASyncContext, batch: int = 8):
    page = context.pages[0]
    agent = AgentT.from_page(page=page, tmp_dir=tmp_dir)
    await page.goto(SiteKey.as_sitelink(sitekey="user"))

    await agent.handle_checkbox()

    for pth in range(1, batch + 1):
        label = await agent.collete()
        labels.add(label)
        print(f"\r>> COLLETE - progress={pth}/{batch} {label=}", end="")
        await page.wait_for_timeout(500)
        fl = page.frame_locator(agent.HOOK_CHALLENGE)
        await fl.locator("//div[@class='refresh button']").click()


async def bytedance():
    malenia = Malenia(user_data_dir=context_dir)
    await malenia.execute(sequence=[collete_datasets], headless=True)

    print(f"\n>> COUNT - {labels=}")

if __name__ == "__main__":
    asyncio.run(bytedance())
