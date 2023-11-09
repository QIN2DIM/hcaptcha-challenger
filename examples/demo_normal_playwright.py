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
from hcaptcha_challenger.agents import AgentT
from hcaptcha_challenger.utils import SiteKey

# Init local-side of the ModelHub
clip_available = True
solver.install(upgrade=True, clip=clip_available)

# Save dataset to current working directory
tmp_dir = Path(__file__).parent.joinpath("tmp_dir")

# sitekey = "58366d97-3e8c-4b57-a679-4a41c8423be3"
sitekey = SiteKey.user_easy


def patch_datalake(modelhub: solver.ModelHub):
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

    agent = AgentT.from_page(page=page, tmp_dir=tmp_dir, self_supervised=clip_available)
    patch_datalake(agent.modelhub)

    await page.goto(SiteKey.as_sitelink(sitekey))

    await agent.handle_checkbox()

    for pth in range(1, times):
        result = await agent.execute()
        probe = list(agent.qr.requester_restricted_answer_set.keys())
        question = agent.qr.requester_question
        print(f">> {pth} - Challenge Result: {result} - {question=} {probe=}")
        match result:
            case agent.status.CHALLENGE_BACKCALL:
                await page.wait_for_timeout(500)
                fl = page.frame_locator(agent.HOOK_CHALLENGE)
                await fl.locator("//div[@class='refresh button']").click()
            case agent.status.CHALLENGE_SUCCESS:
                return


async def bytedance():
    # playwright install chromium --with-deps
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        context = await browser.new_context(
            locale="en-US", record_video_dir=Path("user_data_dir/record")
        )
        await hit_challenge(context)


if __name__ == "__main__":
    asyncio.run(bytedance())
