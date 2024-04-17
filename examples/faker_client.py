# -*- coding: utf-8 -*-
# Time       : 2024/4/1 21:10
# Author     : QIN2DIM
# GitHub     : https://github.com/QIN2DIM
# Description:
from __future__ import annotations

import asyncio
from pathlib import Path

from playwright.async_api import async_playwright, BrowserContext

from hcaptcha_challenger.agents import AgentV
from hcaptcha_challenger.agents import Malenia
from hcaptcha_challenger.utils import SiteKey


# 1. You need to deploy sub-thread tasks and actively run `install(upgrade=True)` every 20 minutes
# 2. You need to make sure to run `install(upgrade=True, clip=True)` before each instantiation
# install(upgrade=True, clip=True)


async def main(headless: bool = False):
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=headless)
        context = await browser.new_context(locale="en-US")
        await Malenia.apply_stealth(context)
        await mime(context)
        await context.close()


async def mime(context: BrowserContext):
    page = await context.new_page()

    agent = AgentV.into_solver(page=page, tmp_dir=Path("tmp_dir"))

    sitekey = SiteKey.user_easy

    if EXECUTION == "challenge":
        sitelink = SiteKey.as_sitelink(sitekey)
        await page.goto(sitelink)
        await agent.ms.click_checkbox()
        await agent.wait_for_challenge()
    elif EXECUTION == "collect":
        await agent.wait_for_collect(sitekey, batch=25)


if __name__ == "__main__":
    EXECUTION = "collect"
    # EXECUTION = "challenge"

    encrypted_resp = asyncio.run(main(headless=False))
