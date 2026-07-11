import asyncio
import json
import os

from playwright.async_api import async_playwright, Page

from hcaptcha_challenger import AgentV, AgentConfig, CaptchaResponse
from hcaptcha_challenger.utils import SiteKey

# Run the agent against any OpenAI-compatible endpoint instead of Gemini.
#
# Works with the official OpenAI API, OpenRouter, and self-hosted engines
# (Ollama, vLLM, SGLang, LM Studio, llama.cpp, TGI, LocalAI).
#
# [IMPORTANT] A vision-capable model is required (e.g. Qwen2-VL, LLaVA, Pixtral).
# Text-only models cannot solve visual challenges.
#
# Install the optional dependency first:
#   pip install hcaptcha-challenger[openai]


def build_local_ollama_config() -> AgentConfig:
    """Example A: a local Ollama server (no authentication)."""
    return AgentConfig(
        CHAT_PROVIDER="openai-compatible",
        OPENAI_BASE_URL="http://localhost:11434/v1",
        # Increase for slow local VLMs on modest hardware.
        OPENAI_TIMEOUT=120,
        # Point every task at your local vision model.
        CHALLENGE_CLASSIFIER_MODEL="qwen2-vl:7b",
        IMAGE_CLASSIFIER_MODEL="qwen2-vl:7b",
        SPATIAL_POINT_REASONER_MODEL="qwen2-vl:7b",
        SPATIAL_PATH_REASONER_MODEL="qwen2-vl:7b",
    )


def build_hosted_openai_config() -> AgentConfig:
    """Example B: a hosted OpenAI-compatible API (key from environment)."""
    return AgentConfig(
        CHAT_PROVIDER="openai-compatible",
        OPENAI_API_KEY=os.getenv("OPENAI_API_KEY", ""),
        # OPENAI_BASE_URL is unset -> the official OpenAI API is used.
        IMAGE_CLASSIFIER_MODEL="gpt-4o",
        SPATIAL_POINT_REASONER_MODEL="gpt-4o",
        SPATIAL_PATH_REASONER_MODEL="gpt-4o",
        CHALLENGE_CLASSIFIER_MODEL="gpt-4o",
    )


async def challenge(page: Page) -> AgentV:
    """Automates the process of solving an hCaptcha challenge."""
    # Swap build_local_ollama_config() for build_hosted_openai_config() to
    # target a hosted API instead of a local server.
    agent_config = build_local_ollama_config()
    agent = AgentV(page=page, agent_config=agent_config)

    await agent.robotic_arm.click_checkbox()
    await agent.wait_for_challenge()

    return agent


async def main():
    async with async_playwright() as p:
        context = await p.chromium.launch_persistent_context(
            user_data_dir="tmp/.cache/user_data", headless=False, locale="en-US"
        )

        page = await context.new_page()

        await page.goto(SiteKey.as_site_link(SiteKey.discord))

        agent: AgentV = await challenge(page)
        if agent.cr_list:
            cr: CaptchaResponse = agent.cr_list[-1]
            print(json.dumps(cr.model_dump(by_alias=True), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())
