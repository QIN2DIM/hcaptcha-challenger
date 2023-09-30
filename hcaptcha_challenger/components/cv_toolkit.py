# -*- coding: utf-8 -*-
# Time       : 2023/9/29 14:19
# Author     : QIN2DIM
# GitHub     : https://github.com/QIN2DIM
# Description:
from typing import Tuple, List

import cv2
import numpy as np
from scipy.spatial.distance import cdist


def limited_radius(img) -> int:
    r = 48 if 512 in img.shape else 32
    return r


def annotate_objects(image_path: str):
    img = cv2.imread(image_path)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    burl = cv2.bilateralFilter(gray, 30, 50, 10)
    canny = cv2.Canny(burl, 15, 200)

    _, thresh = cv2.threshold(canny, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    markers = cv2.watershed(img, markers)
    img[markers == -1] = [255, 0, 0]

    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT, 1, 65, param1=100, param2=40, minRadius=5, maxRadius=65
    )
    if np.all(circles):
        circles = np.uint16(np.around(circles))
        ix = []
        for i in circles[0, :]:
            lr = limited_radius(img)
            if i[2] > lr + 2:
                continue
            i[2] = lr
            ix.append(i)
        return img, ix
    return img, None


def find_unique_object(img: np.ndarray, circles: List[List[int]]) -> Tuple[int, int, int]:
    regions = []
    for x, y, r in circles:
        roi = img[y - r : y + r, x - r : x + r]
        hist = cv2.calcHist([roi], [0, 1, 2], None, [16, 16, 16], [0, 256, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        regions.append(hist)

    dist = cdist(regions, regions, metric="chebyshev")

    i, j = np.unravel_index(np.argmax(dist), dist.shape)
    the_x, the_y = circles[i][:2]

    return the_x, the_y, limited_radius(img)
