# -*- coding: utf-8 -*-
# Time       : 2022-03-01 20:32
# Author     : Bingjie Yan
# Github     : https://github.com/beiyuouo
# Description:
import hashlib
import os
import time
from typing import Optional

import cv2
import numpy as np
import requests
import yaml
from skimage import feature
from skimage.future import graph
from skimage.segmentation import slic


class SKRecognition:
    def __init__(self, path_rainbow: Optional[str] = None):
        self.path_rainbow = "rainbow.yaml" if path_rainbow is None else path_rainbow
        self.flag = "skimage_model"
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
        _rainbow_table = {}

        if os.path.exists(path_rainbow):
            with open(path_rainbow, "r", encoding="utf8") as file:
                stream = yaml.safe_load(file)
            _rainbow_table = stream if isinstance(stream, dict) else {}

        return _rainbow_table

    def match_rainbow(self, img_stream: bytes, rainbow_key: str) -> Optional[bool]:
        """
        # This algorithm is too fast! You need to add some delay before returning the correct result.
        # Without the delay, this solution would have passed the challenge in `0.03s`,
        # which is `humanly' impossible to do.
        :param img_stream:
        :param rainbow_key:
        :return:
        """
        try:
            if self.rainbow_table[rainbow_key]["yes"].get(
                hashlib.md5(img_stream).hexdigest()
            ):
                return True
            if self.rainbow_table[rainbow_key]["bad"].get(
                hashlib.md5(img_stream).hexdigest()
            ):
                return False
        except KeyError:
            pass
        return None

    def solution(self, img_stream, **kwargs) -> bool:
        """Implementation process of solution"""
        raise NotImplementedError


class RiverChallenger(SKRecognition):
    """A fast solution for identifying vertical rivers"""

    def __init__(self, path_rainbow: Optional[str] = None):
        super().__init__(path_rainbow=path_rainbow)

    @staticmethod
    def _weight_mean_color(graph_, src: int, dst: int, n: int):  # noqa
        """Callback to handle merging nodes by recomputing mean color.

        The method expects that the mean color of `dst` is already computed.

        Parameters
        ----------
        graph_ : RAG
            The graph under consideration.
        src, dst : int
            The vertices in `graph` to be merged.
        n : int
            A neighbor of `src` or `dst` or both.

        Returns
        -------
        data : dict
            A dictionary with the `"weight"` attribute set as the absolute
            difference of the mean color between node `dst` and `n`.
        """

        diff = graph_.nodes[dst]["mean color"] - graph_.nodes[n]["mean color"]
        diff = np.linalg.norm(diff)
        return {"weight": diff}

    @staticmethod
    def _merge_mean_color(graph_, src: int, dst: int):
        """Callback called before merging two nodes of a mean color distance graph.

        This method computes the mean color of `dst`.

        Parameters
        ----------
        graph_ : RAG
            The graph under consideration.
        src, dst : int
            The vertices in `graph` to be merged.
        """
        graph_.nodes[dst]["total color"] += graph_.nodes[src]["total color"]
        graph_.nodes[dst]["pixel count"] += graph_.nodes[src]["pixel count"]
        graph_.nodes[dst]["mean color"] = (
            graph_.nodes[dst]["total color"] / graph_.nodes[dst]["pixel count"]
        )

    def solution(self, img_stream, **kwargs) -> bool:  # noqa
        """Implementation process of solution"""
        match_output = self.match_rainbow(img_stream, rainbow_key="vertical river")
        if match_output is not None:
            time.sleep(0.03)
            return match_output

        img_arr = np.frombuffer(img_stream, np.uint8)
        img = cv2.imdecode(img_arr, flags=1)

        img = cv2.pyrMeanShiftFiltering(img, sp=10, sr=40)
        img = cv2.bilateralFilter(img, d=9, sigmaColor=100, sigmaSpace=75)

        labels = slic(img, compactness=30, n_segments=400, start_label=1)
        g = graph.rag_mean_color(img, labels)

        labels2 = graph.merge_hierarchical(
            labels,
            g,
            thresh=35,
            rag_copy=False,
            in_place_merge=True,
            merge_func=self._merge_mean_color,
            weight_func=self._weight_mean_color,
        )

        return len(np.unique(labels2[-1])) >= 3


class DetectionChallenger(SKRecognition):
    """A fast solution for identifying `airplane in the sky flying left`"""

    def __init__(self, path_rainbow: Optional[str] = None):
        super().__init__(path_rainbow=path_rainbow)
        self.sky_threshold = 1800
        self.left_threshold = 30

    @staticmethod
    def _remove_border(img):
        img[:, 1] = 0
        img[:, -2] = 0
        img[1, :] = 0
        img[-2, :] = 0
        return img

    def solution(self, img_stream: bytes, **kwargs) -> bool:
        """Implementation process of solution"""
        match_output = self.match_rainbow(
            img_stream, rainbow_key="airplane in the sky flying left"
        )
        if match_output is not None:
            time.sleep(0.03)
            return match_output

        img_arr = np.frombuffer(img_stream, np.uint8)
        img = cv2.imdecode(img_arr, flags=1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        edges1 = feature.canny(img)
        edges1 = self._remove_border(edges1)

        # on the ground
        if np.count_nonzero(edges1) > self.sky_threshold:
            return False

        min_x = np.min(np.nonzero(edges1), axis=1)[1]
        max_x = np.max(np.nonzero(edges1), axis=1)[1]

        left_nonzero = np.count_nonzero(
            edges1[:, min_x : min(max_x, min_x + self.left_threshold)]
        )
        right_nonzero = np.count_nonzero(
            edges1[:, max(min_x, max_x - self.left_threshold) : max_x]
        )

        # Flying towards the right
        if left_nonzero > right_nonzero:
            return False

        time.sleep(0.25)
        return True
