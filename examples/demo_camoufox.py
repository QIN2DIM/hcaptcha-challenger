import asyncio
import json

from browserforge.fingerprints import Screen
from camoufox import AsyncCamoufox
from playwright.async_api import Page

from hcaptcha_challenger.agent import AgentV, AgentConfig
from hcaptcha_challenger.models import CaptchaResponse
from hcaptcha_challenger.utils import SiteKey


async def challenge(page: Page) -> AgentV:
    """Automates the process of solving an hCaptcha challenge."""
    # [IMPORTANT] Initialize the Agent before triggering hCaptcha
    agent_config = AgentConfig(DISABLE_BEZIER_TRAJECTORY=True)
    agent = AgentV(page=page, agent_config=agent_config)

    # In your real-world workflow, you may need to replace the `click_checkbox()`
    # It may be to click the Login button or the Submit button to trigger challenge
    await agent.robotic_arm.click_checkbox()

    # Wait for the challenge to appear and be ready for solving
    await agent.wait_for_challenge()

    return agent


async def main():
    """
    uv pip install -U hcaptcha-challenger[camoufox]
    Returns:

    """

    async with AsyncCamoufox(
        persistent_context=True,
        user_data_dir="tmp/.cache/camoufox",
        screen=Screen(max_width=1920, max_height=1080),
        humanize=0.5,  # humanize=True,
    ) as browser:
        page = browser.pages[-1] if browser.pages else await browser.new_page()

        await page.goto(SiteKey.as_site_link(SiteKey.user))

        # --- When you encounter hCaptcha in your workflow ---
        agent: AgentV = await challenge(page)

        # Print the last CaptchaResponse
        if agent.cr_list:
            cr: CaptchaResponse = agent.cr_list[-1]
            print(json.dumps(cr.model_dump(by_alias=True), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())
