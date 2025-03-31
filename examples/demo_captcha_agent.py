# -*- coding: utf-8 -*-
# Time       : 2024/4/1 21:10
# Author     : QIN2DIM
# GitHub     : https://github.com/QIN2DIM
# Description:
from __future__ import annotations

import asyncio
import json

from undetected_playwright.async_api import async_playwright, BrowserContext

from hcaptcha_challenger.agent import AgentV, AgentConfig
from hcaptcha_challenger.models import CaptchaResponse
from hcaptcha_challenger.utils import SiteKey


async def challenge(context: BrowserContext):
    """
    Automates the process of solving an hCaptcha challenge.

    This function creates a new browser page, initializes an agent with the necessary
    configuration, navigates to an hCaptcha test page, and attempts to solve the challenge
    by clicking the checkbox and handling the verification process.

    Args:
        context (BrowserContext): The browser context in which to perform the automation.
                                 This is typically created by a Playwright browser instance.

    Returns:
        None

    Note:
        This is an hCaptcha challenge automation. It demonstrates how to interact with
        hCaptcha elements using the agent's robotic arm interface.
    """
    # Create a new page in the provided browser context
    page = await context.new_page()

    # Navigate to the hCaptcha test page using a predefined site key
    # SiteKey.user_easy likely refers to a test/demo hCaptcha with lower difficulty
    await page.goto(SiteKey.as_site_link(SiteKey.discord))
    # await page.goto(SiteKey.as_site_link(SiteKey.user_easy))

    # --- Suppose you encounter hCaptcha in your browser ---

    # Initialize the agent configuration with API key (from parameters or environment)
    agent_config = AgentConfig()

    # Create an agent instance with the page and configuration
    # AgentV appears to be a specialized agent for visual challenges
    agent = AgentV(page=page, agent_config=agent_config)

    # Click the hCaptcha checkbox to initiate the challenge
    # The robotic_arm is an abstraction for performing UI interactions
    await agent.robotic_arm.click_checkbox()

    # Wait for the challenge to appear and be ready for solving
    # This may involve waiting for images to load or instructions to appear
    await agent.wait_for_challenge()

    # Note: The code ends here, suggesting this is part of a larger solution
    # that would continue with challenge solving steps after this point
    if agent.cr_list:
        cr: CaptchaResponse = agent.cr_list[-1]
        print(json.dumps(cr.model_dump(by_alias=True), indent=2, ensure_ascii=False))
        return cr


async def main():
    async with async_playwright() as p:
        context = await p.chromium.launch_persistent_context(
            user_data_dir="tmp/.cache/user_data",
            headless=False,
            # record_video_dir=Path("tmp/.cache/record"),
            # record_video_size=ViewportSize(width=1920, height=1080),
            locale="en-US",
        )
        await challenge(context)
        await context.close()


if __name__ == "__main__":
    encrypted_resp = asyncio.run(main())
