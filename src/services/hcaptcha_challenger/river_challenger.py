#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    :   src\services\hcaptcha_challenger\river_challenger.py
# @Time    :   2022-03-01 20:32:08
# @Author  :   Bingjie Yan
# @Email   :   bj.yan.pa@qq.com
# @License :   Apache License 2.0

import cv2
import numpy as np

import skimage
from skimage.morphology import disk
from skimage.segmentation import watershed, slic, mark_boundaries
from skimage.filters import rank
from skimage.util import img_as_ubyte
from skimage.color import rgb2gray, label2rgb
from skimage.future import graph
from scipy import ndimage as ndi
import matplotlib.pyplot as plt


def _weight_mean_color(graph, src, dst, n):
    """Callback to handle merging nodes by recomputing mean color.

    The method expects that the mean color of `dst` is already computed.

    Parameters
    ----------
    graph : RAG
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

    diff = graph.nodes[dst]['mean color'] - graph.nodes[n]['mean color']
    diff = np.linalg.norm(diff)
    return {'weight': diff}


def merge_mean_color(graph, src, dst):
    """Callback called before merging two nodes of a mean color distance graph.

    This method computes the mean color of `dst`.

    Parameters
    ----------
    graph : RAG
        The graph under consideration.
    src, dst : int
        The vertices in `graph` to be merged.
    """
    graph.nodes[dst]['total color'] += graph.nodes[src]['total color']
    graph.nodes[dst]['pixel count'] += graph.nodes[src]['pixel count']
    graph.nodes[dst]['mean color'] = (graph.nodes[dst]['total color'] /
                                      graph.nodes[dst]['pixel count'])


class RiverChallenger(object):
    def __init__(self) -> None:
        pass

    def challenge(self, img_stream):
        img_arr = np.frombuffer(img_stream, np.uint8)
        img = cv2.imdecode(img_arr, flags=1)
        height, width = img.shape[:2]

        # # filter
        img = cv2.pyrMeanShiftFiltering(img, sp=10, sr=40)
        img = cv2.bilateralFilter(img, d=9, sigmaColor=100, sigmaSpace=75)

        # # enhance brightness
        # img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # img_hsv[:, :, 2] = img_hsv[:, :, 2] * 1.2
        # img = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)

        labels = slic(img, compactness=30, n_segments=400, start_label=1)
        g = graph.rag_mean_color(img, labels)

        labels2 = graph.merge_hierarchical(labels,
                                           g,
                                           thresh=35,
                                           rag_copy=False,
                                           in_place_merge=True,
                                           merge_func=merge_mean_color,
                                           weight_func=_weight_mean_color)

        # view results
        # out = label2rgb(labels2, img, kind='avg', bg_label=0)
        # out = mark_boundaries(out, labels2, (0, 0, 0))
        # skimage.io.imshow(out)
        # skimage.io.show()
        # print(np.unique(labels2[-1]))

        ref_value = len(np.unique(labels2[-1]))
        return ref_value >= 3


if __name__ == '__main__':
    import os
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    result_path = 'result.txt'
    if os.path.exists(result_path):
        os.remove(result_path)

    result_file = open(result_path, 'w')
    # result_file = sys.stdout

    base_path = os.path.join('database', '_challenge')
    list_dirs = os.listdir(base_path)
    for dir in list_dirs:
        print(dir, file=result_file)
        for i in range(1, 10):
            img_filepath = os.path.join(base_path, dir, f'挑战图片{i}.png')

            with open(img_filepath, "rb") as file:
                data = file.read()

            rc = RiverChallenger()
            result = rc.challenge(data)
            print(f'挑战图片{i}.png:{result}', file=result_file)