# -*- coding: utf-8 -*-
# Time       : 2023/10/5 16:23
# Author     : QIN2DIM
# GitHub     : https://github.com/QIN2DIM
# Description:
import shutil
from pathlib import Path

from playwright.async_api import async_playwright

from hcaptcha_challenger.agents import Malenia, AgentT
from hcaptcha_challenger.utils import SiteKey


async def test_normal_instance():
    tmp_dir = Path(__file__).parent.joinpath("tmp_dir_normal_instance")
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

        for pth in range(1, 3):
            result = await agent()
            probe = list(agent.qr.requester_restricted_answer_set.keys())
            question = agent.qr.requester_question
            print(f">> {pth} - Challenge Result: {result} - {question=} {probe=}")
            if result in [agent.status.CHALLENGE_BACKCALL]:
                await page.wait_for_timeout(500)
                fl = page.frame_locator(agent.HOOK_CHALLENGE)
                await fl.locator("//div[@class='refresh button']").click()
            elif result in [agent.status.CHALLENGE_SUCCESS]:
                return
