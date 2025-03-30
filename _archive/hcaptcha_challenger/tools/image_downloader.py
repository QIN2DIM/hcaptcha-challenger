# -*- coding: utf-8 -*-
# Time       : 2023/8/19 17:52
# Author     : QIN2DIM
# GitHub     : https://github.com/QIN2DIM
# Description:
from __future__ import annotations

import asyncio
import hashlib
import shutil
import sys
import time
from abc import ABC, abstractmethod
from contextlib import suppress
from pathlib import Path
from typing import Any, Tuple, List

import httpx
from httpx import AsyncClient
from tenacity import *
from hcaptcha_challenger.models import QuestionResp
from hcaptcha_challenger.constant import INV

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


async def download_challenge_images(
    qr: QuestionResp, label: str, tmp_dir: Path, ignore_examples: bool = False
):
    request_type = qr.request_type
    ks = list(qr.requester_restricted_answer_set.keys())

    for c in INV:
        label = label.replace(c, "")
    label = label.strip()

    if len(ks) > 0:
        typed_dir = tmp_dir.joinpath(request_type, label, ks[0])
    else:
        typed_dir = tmp_dir.joinpath(request_type, label)
    typed_dir.mkdir(parents=True, exist_ok=True)

    ciri = Cirilla()
    container = []
    tasks = []
    for i, tk in enumerate(qr.tasklist):
        challenge_img_path = typed_dir.joinpath(f"{time.time()}.{i}.png")
        context = (challenge_img_path, tk.datapoint_uri)
        container.append(context)
        tasks.append(asyncio.create_task(ciri.elder_blood(context)))

    examples = []
    if not ignore_examples:
        with suppress(Exception):
            for i, uri in enumerate(qr.requester_question_example):
                example_img_path = typed_dir.joinpath(f"{time.time()}.exp.{i}.png")
                context = (example_img_path, uri)
                examples.append(context)
                tasks.append(asyncio.create_task(ciri.elder_blood(context)))

    await asyncio.gather(*tasks)

    # Optional deduplication
    _img_paths = []
    for src, _ in container:
        cache = src.read_bytes()
        dst = typed_dir.joinpath(f"{hashlib.md5(cache).hexdigest()}.png")
        shutil.move(src, dst)
        _img_paths.append(dst)

    # Optional deduplication
    _example_paths = []
    if examples:
        for src, _ in examples:
            cache = src.read_bytes()
            dst = typed_dir.joinpath(f"{hashlib.md5(cache).hexdigest()}.png")
            shutil.move(src, dst)
            _example_paths.append(dst)

    return _img_paths, _example_paths
