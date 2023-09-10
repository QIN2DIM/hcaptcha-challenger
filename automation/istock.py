# -*- coding: utf-8 -*-
# Time       : 2022/8/5 8:45
# Author     : QIN2DIM
# Github     : https://github.com/QIN2DIM
# Description:
import asyncio
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Set
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup
from httpx import AsyncClient
from loguru import logger

UNDEFINED = "undefined"

MediaType = Literal["photography", "illustration", "illustration&assetfiletype=eps", "undefined"]
Orientations = Literal["square", "vertical", "horizontal", "panoramicvertical", "panoramichorizontal", "undefined"]
NumberOfPeople = Literal["none", "one", "two", "group", "undefined"]


@dataclass
class Istock:
    phrase: str
    """
    Required. Keywords to more_like_this
    """

    mediatype: MediaType = "photography"
    """
    Optional. Default "photography". Choose from `MediaType()`.
    """

    number_of_people: NumberOfPeople = "none"
    """
    Optional. Default "none". Choose from `NumberOfPeople()`.
    """

    orientations: Orientations = "undefined"
    """
    Optional. Default "undefined". Choose from `Orientations()`.
    """

    pages: int = 5
    """
    Default 5. Value interval `pages∈[1, 20]`
    """

    flag: bool = True
    """
    Optional. Default True. File storage path.
    IF True --(*.jpg)--> `./istock_dataset/phrase/`
    ELSE --(*.jpg)--> `./istock_dataset/undefined/`
    """

    tmp_dir: Path = Path(__file__).parent.joinpath("tmp_dir")
    store_dir: Path = field(default=Path)
    """
    Optional. Default `./istock_dataset/`. Image database root directory.
    """

    cases_name: Set[str] = field(default_factory=set)
    work_queue: asyncio.Queue[str] | None = None
    client: AsyncClient | None = None
    api = "https://www.istockphoto.com/search/2/image"

    def __post_init__(self):
        logger.debug(f"Container preload - phrase={self.phrase}")

        self.work_queue = asyncio.Queue()

        self.store_dir = self.tmp_dir.joinpath("istock_tmp", self.phrase)
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self.cases_name = set(os.listdir(self.store_dir))

        headers = {
            "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/116.0.0.0 Safari/537.36 Edg/116.0.1938.76"
        }
        self.client = AsyncClient(headers=headers)

    @classmethod
    def from_phrase(cls, phrase: str, **kwargs):
        return cls(phrase=phrase.strip(), **kwargs)

    async def preload(self):
        p = urlparse(self.api)
        if p.path == "/search/2/image" and "colorsimilarityassetid" in p.query:
            params = f"&phrase={self.phrase}"
        else:
            params = f"?phrase={self.phrase}"

        # The others query
        if self.mediatype != UNDEFINED:
            params += f"&mediatype={self.mediatype}"
        if self.number_of_people != UNDEFINED:
            params += f"&numberofpeople={self.number_of_people}"
        if self.orientations != UNDEFINED:
            params += f"&orientations={self.orientations}"

        img_index_urls = [f"{self.api}{params}&page={i}" for i in range(1, self.pages + 1)]
        logger.info(f"preload - size={len(img_index_urls)}")

        return img_index_urls

    async def adaptor(self):
        while not self.work_queue.empty():
            context = self.work_queue.get_nowait()
            if not context or not isinstance(context, str):
                print(f"Drop url - context={context}")
            elif context.startswith("https://media.istockphoto.com/"):
                await self.download_image(context)

    async def get_image_urls(self, url: str):
        """Get download links for all images in the page"""
        try:
            response = await self.client.get(url)
        except httpx.ConnectTimeout:
            pass
        else:
            soup = BeautifulSoup(response.text, "html.parser")
            gallery = soup.find("div", attrs={"data-testid": "gallery-items-container"})
            if gallery:
                img_tags = gallery.find_all("img")
                for tag in img_tags:
                    self.work_queue.put_nowait(tag["src"])

    async def download_image(self, url: str):
        """Download thumbnail"""
        istock_id = f"{urlparse(url).path.split('/')[2]}"
        img_path = self.store_dir.joinpath(f"{istock_id}.jpg")
        if img_path.name not in self.cases_name:
            res = await self.client.get(url, timeout=30)
            img_path.write_bytes(res.content)

    async def more_like_this(
            self, istock_id: str | int, similar: Literal["content", "color"] = "content"
    ):
        """

        :param istock_id:
        :param similar:
        :return:
        """
        similar_match = {
            "content": f"https://www.istockphoto.com/search/more-like-this/{istock_id}",
            "color": f"https://www.istockphoto.com/search/2/image?colorsimilarityassetid={istock_id}",
        }
        self.api = similar_match[similar]
        response = await self.client.get(self.api)
        if not response:
            logger.error(f"Could not find source image in istock by the istock_id({istock_id})")
            raise

        return self

    async def mining(self):
        urls = await self.preload()

        logger.info("matching index")
        await asyncio.gather(*[self.get_image_urls(url) for url in urls])

        logger.info("running adaptor...")
        await asyncio.gather(*[self.adaptor() for _ in range(32)])


if __name__ == '__main__':
    istock = Istock.from_phrase("squirrel")
    istock.pages = 4
    asyncio.run(istock.mining())
