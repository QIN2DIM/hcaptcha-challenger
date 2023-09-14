# -*- coding: utf-8 -*-
# Time       : 2023/9/14 13:32
# Author     : QIN2DIM
# GitHub     : https://github.com/QIN2DIM
# Description:
from __future__ import annotations

import pytest
from playwright.async_api import async_playwright

import hcaptcha_challenger as solver
from hcaptcha_challenger.utils import SiteKey

# Init local-side of the ModelHub
solver.install(upgrade=True, flush_yolo=False)


@pytest.mark.parametrize("sitekey", [SiteKey.epic, SiteKey.discord, SiteKey.user_easy])
@pytest.mark.parametrize("times", [3])
async def test_normal_playwright(sitekey: str, times: int):
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(locale="en-US")
        page = await context.new_page()

        agent = solver.AgentT.from_page(page=page)
        await page.goto(SiteKey.as_sitelink(sitekey))

        await agent.handle_checkbox()

        state = agent.status.CHALLENGE_BACKCALL
        msg = ""
        for pth in range(1, times):
            result = await agent()
            state = result
            if result in [agent.status.CHALLENGE_SUCCESS]:
                return
            if result in [agent.status.CHALLENGE_RETRY]:
                continue
            probe = list(agent.qr.requester_restricted_answer_set.keys())
            msg = f"{result=} label={agent._label} {probe=}"
        assert state == agent.status.CHALLENGE_BACKCALL, msg
