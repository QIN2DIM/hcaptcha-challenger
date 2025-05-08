import asyncio
from pathlib import Path

from playwright.async_api import async_playwright

from hcaptcha_challenger import Collector, CollectorConfig
from hcaptcha_challenger.utils import SiteKey

dataset_dir = Path("tmp/.cache/dataset").resolve()
site_key = SiteKey.epic


async def launch_collector():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        context = await browser.new_context(locale="en-US")
        page = await context.new_page()

        collector_config = CollectorConfig(dataset_dir=dataset_dir, site_key=site_key)
        collector = Collector(page, collector_config)

        await collector.launch()


if __name__ == '__main__':
    asyncio.run(launch_collector())
