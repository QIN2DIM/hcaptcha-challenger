# -*- coding: utf-8 -*-
# Time       : 2023/8/19 17:52
# Author     : QIN2DIM
# GitHub     : https://github.com/QIN2DIM
# Description:
from __future__ import annotations

import asyncio
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Tuple, List

import httpx
from httpx import AsyncClient
from tenacity import *

DownloadList = List[Tuple[Path, str]]


class AshFramework(ABC):
    def __init__(self, container: DownloadList):
        self.container = container

    @classmethod
    def from_container(cls, container):
        if sys.platform.startswith("win") or "cygwin" in sys.platform:
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        else:
            asyncio.set_event_loop(asyncio.new_event_loop())
        return cls(container=container)

    @abstractmethod
    async def control_driver(self, context: Any, client: AsyncClient):
        raise NotImplementedError

    async def subvert(self):
        if not self.container:
            return
        async with AsyncClient() as client:
            task_list = [self.control_driver(context, client) for context in self.container]
            await asyncio.gather(*task_list)

    def execute(self):
        asyncio.run(self.subvert())


class ImageDownloader(AshFramework):
    async def control_driver(self, context: Any, client: AsyncClient):
        (img_path, url) = context
        resp = await client.get(url)
        img_path.write_bytes(resp.content)


def download_images(container: DownloadList):
    """
    Download Challenge Image

    ### hcaptcha has a challenge duration limit

    If the page element is not manipulated for a period of time,
    the <iframe> box will disappear and the previously acquired Element Locator will be out of date.
    Need to use some modern methods to shorten the time of `getting the dataset` as much as possible.

    ### Solution

    1. Coroutine Downloader
      Use the coroutine-based method to _pull the image to the local, the best practice (this method).
      In the case of poor network, _pull efficiency is at least 10 times faster than traversal download.

    2. Screen cut
      There is some difficulty in coding.
      Directly intercept nine pictures of the target area, and use the tool function to cut and identify them.
      Need to weave the locator index yourself.

    :param container:
    :return:
    """
    ImageDownloader(container).execute()


class Cirilla:
    def __init__(self):
        self.client = AsyncClient(http2=True, timeout=3)

    @retry(
        retry=retry_if_exception_type(httpx.RequestError),
        wait=wait_random_exponential(multiplier=1, max=60),
        stop=(stop_after_delay(60) | stop_after_attempt(10)),
    )
    async def elder_blood(self, context):
        img_path, url = context

        resp = await self.client.get(url)
        img_path.write_bytes(resp.content)


def common_download(container: DownloadList):
    for img_path, url in container:
        resp = httpx.get(url)
        img_path.write_bytes(resp.content)
