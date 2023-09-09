# -*- coding: utf-8 -*-
# Time       : 2023/8/31 20:54
# Author     : QIN2DIM
# GitHub     : https://github.com/QIN2DIM
# Description:
import asyncio
from pathlib import Path
from typing import List

import httpx
from loguru import logger
from playwright.async_api import BrowserContext as ASyncContext, async_playwright

from hcaptcha_challenger import AgentT, Malenia
from hcaptcha_challenger.utils import SiteKey

# Save dataset to current working directory
tmp_dir = Path(__file__).parent.joinpath("tmp_dir")
user_data_dir = Path(__file__).parent.joinpath("user_data_dir")
context_dir = user_data_dir.joinpath("context")

labels = set()

per_times: int = 20
loop_times: int = 3
sitelinks: List[str] = [
    SiteKey.as_sitelink(sitekey=SiteKey.user_easy),
    # SiteKey.as_sitelink(sitekey=SiteKey.discord),
    # SiteKey.as_sitelink(sitekey=SiteKey.hcaptcha),
    # SiteKey.as_sitelink(sitekey="eb932362-438e-43b4-9373-141064402110")
]


@logger.catch
async def collete_datasets(context: ASyncContext, sitelink: str):
    page = await context.new_page()
    agent = AgentT.from_page(page=page, tmp_dir=tmp_dir)

    await page.goto(sitelink)

    await agent.handle_checkbox()

    for pth in range(1, per_times + 1):
        try:
            label = await agent.collect()
            labels.add(label)
            print(f">> COLLETE - progress={pth}/{per_times} {label=}")
        except (httpx.HTTPError, httpx.ConnectTimeout) as err:
            logger.warning(f"Collection speed is too fast", reason=err)
            await page.wait_for_timeout(500)
        except FileNotFoundError:
            pass
        except Exception as err:
            print(err)

        await page.wait_for_timeout(500)
        fl = page.frame_locator(agent.HOOK_CHALLENGE)
        await fl.locator("//div[@class='refresh button']").click()


async def bytedance():
    malenia = Malenia(user_data_dir=context_dir)
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(locale="en-US")
        await malenia.apply_stealth(context)
        for sitelink in sitelinks * loop_times:
            await collete_datasets(context, sitelink)
            print(f"\n>> COUNT - {labels=}")
        await context.close()

if __name__ == "__main__":
    asyncio.run(bytedance())
