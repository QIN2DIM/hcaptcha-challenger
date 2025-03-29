# -*- coding: utf-8 -*-
# Time       : 2024/4/1 21:10
# Author     : QIN2DIM
# GitHub     : https://github.com/QIN2DIM
# Description:
from __future__ import annotations

import asyncio
from pathlib import Path

from undetected_playwright.async_api import async_playwright, BrowserContext

from hcaptcha_challenger.agent import AgentV
from hcaptcha_challenger.utils import SiteKey


async def mime(context: BrowserContext):
    page = await context.new_page()

    agent = AgentV(page=page, tmp_dir=Path("tmp_dir"))

    site_key = SiteKey.user_easy

    if EXECUTION == "challenge":
        await page.goto(SiteKey.as_sitelink(site_key))
        await agent.robotic_arm.click_checkbox()
        await agent.wait_for_challenge()
    elif EXECUTION == "collect":
        await agent.wait_for_collect(site_key, batch=2)


async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        context = await browser.new_context(locale="en-US")
        await mime(context)

        await context.close()


if __name__ == "__main__":
    # EXECUTION = "collect"
    EXECUTION = "challenge"

    encrypted_resp = asyncio.run(main())
