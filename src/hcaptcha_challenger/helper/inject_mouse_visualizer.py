from pathlib import Path
from typing import Union

from playwright.async_api import Page as AsyncPage
from playwright.sync_api import Page as SyncPage
from undetected_playwright.async_api import Page as UAsyncPage
from undetected_playwright.sync_api import Page as USyncPage

js_path = Path(__file__).parent.joinpath("assets", "scripts", "mouse_visualizer.js")
script = js_path.read_text(encoding="utf8")


async def inject_mouse_visualizer_global_async(page: AsyncPage):
    """
    Inject mouse position visualization asynchronously in the Playwright page.

    Args:
        page: Playwright Asynchronous Page Object
    """
    await page.evaluate(expression=script)


def inject_mouse_visualizer_global_sync(page: SyncPage):
    """
    Synchronously inject mouse position visualizations in the Playwright page.

    Args:
        page: Playwright Synchronize Page Objects
    """
    page.evaluate(script)


async def inject_mouse_visualizer_global(page: Union[SyncPage, AsyncPage, USyncPage, UAsyncPage]):
    """
    Inject mouse position visualizations into the Playwright page, supporting synchronous and asynchronous APIs.

    Args:
        page: Playwright Page object, can be synchronous or asynchronous
    """
    if isinstance(page, AsyncPage) or isinstance(page, UAsyncPage):
        await inject_mouse_visualizer_global_async(page)
    else:
        inject_mouse_visualizer_global_sync(page)
