# Documentation

## Get started

### Introduction

hCaptcha Challenger (v0.13.0+) leverages the Spatial Chain-of-Thought capabilities of large language models to build an Agentic Workflow, enabling agents to follow instructions and complete general spatial visual tasks without additional training or fine-tuning.

The `Agent` controls browser pages through playwright. In your workflow, the Agent is initialized with the `page` object you pass in, allowing the Agent to take over interactions with the current page. You can implement two independent operations through the `Agent`: `click_checkbox` and `wait_for_challenge`.

### Installation

```bash
uv pip install hcaptcha-challenger
```

### Quickstart

This is an hCaptcha challenge automation. It demonstrates how to interact with hCaptcha elements using the agent's robotic arm interface.

It's worth noting that the Agent only manipulates web pages through the Page object. Therefore, the Agent can run on any "patcher" built on playwright. In other words, any browser that playwright can launch can be used to run the Agent in this manner.

```python
import asyncio
import json

from playwright.async_api import async_playwright, Page

from hcaptcha_challenger.agent import AgentV, AgentConfig
from hcaptcha_challenger.models import CaptchaResponse
from hcaptcha_challenger.utils import SiteKey


async def challenge(page: Page):
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
        browser = await p.chromium.launch(headless=False)
        context = await browser.new_context()

        # Create a new page in the provided browser context
        page = await context.new_page()

        # Navigate to the hCaptcha test page using a predefined site key
        # SiteKey.user_easy likely refers to a test/demo hCaptcha with lower difficulty
        # await page.goto(SiteKey.as_site_link(SiteKey.discord))
        await page.goto(SiteKey.as_site_link(SiteKey.user_easy))

        # --- Suppose you encounter hCaptcha in your browser ---
        await challenge(page)


if __name__ == "__main__":
    encrypted_resp = asyncio.run(main())

```

## Gallery

![image-20250402235820929](assets/image-20250402235820929.png)

### Image Label Binary

<video src="assets/429154580-c2cea4e0-82f4-466f-8c7a-20f8ea63732c.mp4"></video>
