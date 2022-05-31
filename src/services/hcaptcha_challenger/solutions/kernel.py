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
    def __init__(self, name: str, path_rainbow: str = None):
        self.path_rainbow = "rainbow.yaml" if path_rainbow is None else path_rainbow
        self.flag = name
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
            with requests.get(rainbow_obj["src"], stream=True) as response, open(rainbow_obj["path"], "wb") as file:
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

    @staticmethod
    def download_model_(dir_model, path_model, model_src, model_name):
        """Download the de-stylized binary classification model"""
        if os.path.exists(path_model):
            return
        if not os.path.exists(dir_model):
            os.mkdir(dir_model)

        if not model_src.lower().startswith("http"):
            raise ValueError from None

        print(f"Downloading {model_name} from {model_src}")
        with requests.get(model_src, stream=True) as response, open(path_model, "wb") as file:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    file.write(chunk)

    def solution(self, img_stream, **kwargs) -> bool:
        """Implementation process of solution"""
        raise NotImplementedError

    def solution_dev(self, src_dir: str, **kwargs):
        if not os.path.exists(src_dir):
            return
        _suffix = ".png"
        for _prefix, _, files in os.walk(src_dir):
            for filename in files:
                if not filename.endswith(_suffix):
                    continue
                path_img = os.path.join(_prefix, filename)
                with open(path_img, "rb") as file:
                    yield path_img, self.solution(file.read(), **kwargs)


if __name__ == "__main__":
    rr = r"E:\_GithubProjects\Sources\hcaptcha-challenger"
    sd = Solutions("cool")
    for path_, resp in sd.solution_dev(rr):
        input(f"{path_=} {resp=}")
