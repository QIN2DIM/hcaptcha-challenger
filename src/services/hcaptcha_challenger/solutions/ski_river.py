# -*- coding: utf-8 -*-
# Time       : 2022-03-01 20:32
# Author     : Bingjie Yan
# Github     : https://github.com/beiyuouo
# Description:
import cv2
import numpy as np
from skimage.future import graph
from skimage.segmentation import slic


class RiverChallenger:
    """A fast solution for identifying vertical rivers"""

    def __init__(self):
        self.flag = "skimage_model"

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
    def _merge_mean_color(graph_, src, dst):
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
