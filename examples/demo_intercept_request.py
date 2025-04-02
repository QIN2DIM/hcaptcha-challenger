from __future__ import annotations

import asyncio
import json

from undetected_playwright.async_api import async_playwright, BrowserContext

from hcaptcha_challenger.agent import AgentV, AgentConfig
from hcaptcha_challenger.models import CaptchaResponse


# from hcaptcha_challenger.utils import SiteKey


async def _solve(context: BrowserContext, url: str, sitekey: str):

    page = await context.new_page()

    async def intercept_request(route):
        url = route.request.url
        if "hcaptcha.com" in url:
            await route.continue_()  # Allow hCaptcha to load normally
            return  # Prevent further execution

        # Fulfill other requests with custom content
        await route.fulfill(
            status=200,
            content_type="text/html",
            body=f"""
            <html>
                <head>
                    <title>Custom Discord</title>
                    <script src="https://js.hcaptcha.com/1/api.js" async defer></script>
                </head>
                <body style="background-color: black; color: white; text-align: center;">
                    <h1>Welcome to Custom Discord!</h1>
                    <p>Verify you are human:</p>
                    <form action="#" method="POST">
                        <div class="h-captcha" data-sitekey="{sitekey}"></div>
                        <br>
                        <button type="submit">Submit</button>
                    </form>
                </body>
            </html>
            """,
        )

    await page.route("**/*", intercept_request)

    await page.goto(url)

    agent_config = AgentConfig()

    agent = AgentV(page=page, agent_config=agent_config)

    await agent.robotic_arm.click_checkbox()

    await agent.wait_for_challenge()

    if agent.cr_list:
        cr: CaptchaResponse = agent.cr_list[-1]
        print(json.dumps(cr.model_dump(by_alias=True), indent=2, ensure_ascii=False))
        return cr


async def solvehCaptcha(
    url: str = "https://discord.com", sitekey: str = "4c672d35-0701-42b2-88c3-78380b0db560"
):
    async with async_playwright() as p:
        context = await p.chromium.launch_persistent_context(
            user_data_dir="tmp/.cache/user_data",
            headless=False,
            # record_video_dir=Path("tmp/.cache/record"),
            # record_video_size=ViewportSize(width=1920, height=1080),
            locale="en-US",
        )
        await _solve(context, url, sitekey)
        await context.close()


if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    loop.run_until_complete(solvehCaptcha())
    loop.close()
