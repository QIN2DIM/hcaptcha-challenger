# -*- coding: utf-8 -*-
# Time       : 2024/4/1 21:10
# Author     : QIN2DIM
# GitHub     : https://github.com/QIN2DIM
# Description:
from __future__ import annotations

import asyncio
import os
from pathlib import Path

from undetected_playwright.async_api import async_playwright, BrowserContext, ViewportSize

from hcaptcha_challenger.agent import AgentV, AgentConfig
from hcaptcha_challenger.utils import SiteKey


async def mime(context: BrowserContext):
    page = await context.new_page()

    agent_config = AgentConfig(GEMINI_API_KEY=os.environ["GEMINI_API_KEY"])
    agent = AgentV(page=page, agent_config=agent_config)

    await page.goto(SiteKey.as_sitelink(SiteKey.user_easy))

    await agent.robotic_arm.click_checkbox()

    await agent.wait_for_challenge()


async def main():
    async with async_playwright() as p:
        context = await p.chromium.launch_persistent_context(
            user_data_dir="tmp/.cache/user_data",
            headless=False,
            record_video_dir=Path("tmp/.cache/record"),
            record_video_size=ViewportSize(width=1920, height=1080),
            locale="en-US",
        )
        await mime(context)
        await context.close()


if __name__ == "__main__":
    encrypted_resp = asyncio.run(main())
