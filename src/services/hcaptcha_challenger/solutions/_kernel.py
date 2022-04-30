# -*- coding: utf-8 -*-
# Time       : 2022/4/30 22:34
# Author     : QIN2DIM
# Github     : https://github.com/QIN2DIM
# Description:
import hashlib
import os
from typing import Optional

import requests
import yaml


class Solutions:
    def __init__(self, flag: str, path_rainbow: str):
        self.path_rainbow = "rainbow.yaml" if path_rainbow is None else path_rainbow
        self.flag = flag
        self.rainbow_table = self.build_rainbow(path_rainbow=self.path_rainbow)

    @staticmethod
    def sync_rainbow(path_rainbow: str, convert: Optional[bool] = False):
        """
        同步强化彩虹表
        :param path_rainbow:
        :param convert: 强制同步
        :return:
        """
        rainbow_obj = {
            "name": "rainbow_table",
            "path": path_rainbow,
            "src": "https://github.com/QIN2DIM/hcaptcha-challenger/releases/download/model/rainbow.yaml",
        }

        if convert or not os.path.exists(rainbow_obj["path"]):
            print(f"Downloading {rainbow_obj['name']} from {rainbow_obj['src']}")
            with requests.get(rainbow_obj["src"], stream=True) as response, open(
                rainbow_obj["path"], "wb"
            ) as file:
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        file.write(chunk)

    @staticmethod
    def build_rainbow(path_rainbow: str) -> Optional[dict]:
        """

        :param path_rainbow:
        :return:
        """
        _rainbow_table = {}

        if os.path.exists(path_rainbow):
            with open(path_rainbow, "r", encoding="utf8") as file:
                stream = yaml.safe_load(file)
            _rainbow_table = stream if isinstance(stream, dict) else {}

        return _rainbow_table

    def match_rainbow(self, img_stream: bytes, rainbow_key: str) -> Optional[bool]:
        """

        :param img_stream:
        :param rainbow_key:
        :return:
        """
        try:
            if self.rainbow_table[rainbow_key]["yes"].get(hashlib.md5(img_stream).hexdigest()):
                return True
            if self.rainbow_table[rainbow_key]["bad"].get(hashlib.md5(img_stream).hexdigest()):
                return False
        except KeyError:
            pass
        return None

    def solution(self, img_stream, **kwargs) -> bool:
        """Implementation process of solution"""
        raise NotImplementedError
