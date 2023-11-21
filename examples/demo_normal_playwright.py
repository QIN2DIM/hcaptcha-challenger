# -*- coding: utf-8 -*-
# Time       : 2023/9/2 3:30
# Author     : QIN2DIM
# GitHub     : https://github.com/QIN2DIM
# Description:
from __future__ import annotations

import asyncio
from pathlib import Path

from loguru import logger
from playwright.async_api import BrowserContext as ASyncContext, async_playwright, Page

from hcaptcha_challenger import ModelHub, install
from hcaptcha_challenger.agents import AgentT, Malenia
from hcaptcha_challenger.utils import SiteKey

# sitekey = "58366d97-3e8c-4b57-a679-4a41c8423be3"
# sitekey = "4c672d35-0701-42b2-88c3-78380b0db560"
sitekey = SiteKey.user_easy


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


def prelude(page: Page) -> AgentT:
    # 1. You need to deploy sub-thread tasks and actively run `install(upgrade=True)` every 20 minutes
    # 2. You need to make sure to run `install(upgrade=True, clip=True)` before each instantiation
    install(upgrade=True, clip=True)

    modelhub = ModelHub.from_github_repo()
    modelhub.parse_objects()

    # Make arbitrary pre-modifications to modelhub, which is very useful for CLIP models
    patch_modelhub(modelhub)

    agent = AgentT.from_page(
        # page, the control handle of the Playwright Page
        page=page,
        # modelhub, Register modelhub externally, and the agent can patch custom configurations
        modelhub=modelhub,
        # tmp_dir, Mount the cache directory to the current working folder
        tmp_dir=Path(__file__).parent.joinpath("tmp_dir"),
        # clip, Enable CLIP zero-shot image classification method
        clip=True,
    )

    return agent


async def hit_challenge(context: ASyncContext, times: int = 8):
    page = await context.new_page()

    agent = prelude(page)

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
