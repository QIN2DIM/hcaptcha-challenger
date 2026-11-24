# -*- coding: utf-8 -*-
"""
PoC: solve hCaptcha di test page resmi (accounts.hcaptcha.com/demo)
menggunakan provider OpenAI-compatible (9router) + bzd/deepseek-v4.1-flash-1017.

Cara pakai:
    source .venv/bin/activate
    python examples/poc_9router_solve.py [sitekey]
"""

import asyncio
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from loguru import logger
from playwright.async_api import async_playwright

from hcaptcha_challenger.agent import AgentV, AgentConfig
from hcaptcha_challenger.models import CaptchaResponse
from hcaptcha_challenger.utils import SiteKey


async def main() -> int:
    site_key = sys.argv[1] if len(sys.argv) > 1 else SiteKey.user_easy
    logger.info(f"PoC hCaptcha solve | sitekey={site_key}")

    agent_config = AgentConfig()
    logger.info(
        f"backend={agent_config.AGENT_BACKEND} style={agent_config.OPENAI_COMPAT_API_STYLE} "
        f"base_url={agent_config.OPENAI_COMPAT_BASE_URL} model={agent_config.OPENAI_COMPAT_MODEL}"
    )

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            locale="en-US",
            viewport={"width": 1280, "height": 800},
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/159.0.0.0 Safari/537.36"
            ),
        )
        page = await context.new_page()
        await page.goto(SiteKey.as_site_link(site_key), wait_until="domcontentloaded")

        agent = AgentV(page=page, agent_config=agent_config)
        await agent.robotic_arm.click_checkbox()
        signal = await agent.wait_for_challenge()

        logger.info(f"challenge signal: {signal}")
        if agent.cr_list:
            cr: CaptchaResponse = agent.cr_list[-1]
            print(json.dumps(cr.model_dump(by_alias=True), indent=2, ensure_ascii=False))
            await asyncio.sleep(3)

        await browser.close()
        return 0 if signal.value == "success" else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
