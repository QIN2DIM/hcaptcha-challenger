# -*- coding: utf-8 -*-
# Time       : 2023/9/14 1:56
# Author     : QIN2DIM
# GitHub     : https://github.com/QIN2DIM
# Description:
import shutil
from pathlib import Path

from playwright.async_api import async_playwright

from hcaptcha_challenger.agents import Malenia, AgentT
from hcaptcha_challenger.utils import SiteKey


async def _downloader():
    tmp_dir = Path(__file__).parent.joinpath("tmp_dir2")
    shutil.rmtree(tmp_dir, ignore_errors=True)

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(locale="en-US")
        await Malenia.apply_stealth(context)
        page = await context.new_page()

        agent = AgentT.from_page(page=page, tmp_dir=tmp_dir)

        sitelink = SiteKey.as_sitelink(SiteKey.user_easy)
        await page.goto(sitelink)

        await agent.handle_checkbox()

        await agent.collect()

        assert len(agent.img_paths) in [1, 2, 3, 9, 18]

        for img_path in agent.img_paths:
            assert img_path.stat().st_size
