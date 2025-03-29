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
    在 Playwright 页面中异步注入鼠标位置可视化效果。

    Args:
        page: Playwright 异步 Page 对象
    """
    await page.evaluate(expression=script)


def inject_mouse_visualizer_global_sync(page: SyncPage):
    """
    在 Playwright 页面中同步注入鼠标位置可视化效果。

    Args:
        page: Playwright 同步 Page 对象
    """
    page.evaluate(script)


async def inject_mouse_visualizer_global(page: Union[SyncPage, AsyncPage, USyncPage, UAsyncPage]):
    """
    在 Playwright 页面中注入鼠标位置可视化效果，支持同步和异步 API。

    Args:
        page: Playwright Page 对象，可以是同步或异步的
    """
    if isinstance(page, AsyncPage) or isinstance(page, UAsyncPage):
        await inject_mouse_visualizer_global_async(page)
    else:
        inject_mouse_visualizer_global_sync(page)
