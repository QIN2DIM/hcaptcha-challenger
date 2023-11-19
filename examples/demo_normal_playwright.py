# -*- coding: utf-8 -*-
# Time       : 2023/9/2 3:30
# Author     : QIN2DIM
# GitHub     : https://github.com/QIN2DIM
# Description:
from __future__ import annotations

import asyncio
from pathlib import Path

from loguru import logger
from playwright.async_api import BrowserContext as ASyncContext, async_playwright

import hcaptcha_challenger as solver
from hcaptcha_challenger.agents import AgentT, Malenia
from hcaptcha_challenger.utils import SiteKey

# Init local-side of the ModelHub
solver.install(upgrade=True, clip=True)

# Save dataset to current working directory
tmp_dir = Path(__file__).parent.joinpath("tmp_dir")

# sitekey = "58366d97-3e8c-4b57-a679-4a41c8423be3"
# sitekey = "4c672d35-0701-42b2-88c3-78380b0db560"
sitekey = SiteKey.user_easy


def patch_datalake(modelhub: solver.ModelHub):
    """
    for patching CLIP prompt
    :param modelhub:
    :return:
    """
    datalake_post = {
        "animal": {
            "positive_labels": ["animal", "bird"],
            "negative_labels": ["cables", "forklift", "boat"],
        },
        solver.handle("Select all cats."): {"positive_labels": ["cat"], "negative_labels": ["dog"]},
    }
    for prompt_, serialized_binary in datalake_post.items():
        dl = solver.DataLake.from_serialized(serialized_binary)
        modelhub.datalake[prompt_] = dl


@logger.catch
async def hit_challenge(context: ASyncContext, times: int = 8):
    page = await context.new_page()

    agent = AgentT.from_page(page=page, tmp_dir=tmp_dir, self_supervised=True)
    patch_datalake(agent.modelhub)

    url = SiteKey.as_sitelink(sitekey)
    await page.goto(url)
    logger.info("startup sitelink", url=url)

    await agent.handle_checkbox()

    for pth in range(1, times):
        # Handle challenge
        result = await agent.execute()
        if not agent.qr:
            return

        # Post-processing
        match result:
            case agent.status.CHALLENGE_BACKCALL | agent.status.CHALLENGE_RETRY:
                logger.warning(f"retry", pth=pth, ash=agent.ash)
                await page.wait_for_timeout(500)
                fl = page.frame_locator(agent.HOOK_CHALLENGE)
                await fl.locator("//div[@class='refresh button']").click()
            case agent.status.CHALLENGE_SUCCESS:
                logger.success(f"task done", pth=pth, ash=agent.ash)
                rqdata_path = agent.export_rq()
                print(f"\n>> View RQdata path={rqdata_path}")
                return


@logger.catch
async def bytedance(undetected: bool = False):
    # playwright install chromium --with-deps
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        context = await browser.new_context(
            locale="en-US", record_video_dir=Path("user_data_dir/record")
        )
        if undetected:
            await Malenia.apply_stealth(context)

        await hit_challenge(context)


if __name__ == "__main__":
    asyncio.run(bytedance())
