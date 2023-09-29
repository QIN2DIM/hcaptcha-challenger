# -*- coding: utf-8 -*-
# Time       : 2023/9/29 14:19
# Author     : QIN2DIM
# GitHub     : https://github.com/QIN2DIM
# Description:
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
from scipy.spatial.distance import cdist


def radius(img) -> int:
    r = 48 if 512 in img.shape else 32
    return r


def annotate_objects(image_path: str) -> tuple:
    img = cv2.imread(image_path)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    burl = cv2.bilateralFilter(gray, 30, 50, 10)
    canny = cv2.Canny(burl, 15, 100)

    # 分水岭算法
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

    # Hough 圆检测
    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT, 1, 47, param1=100, param2=40, minRadius=5, maxRadius=60
    )
    circles = np.uint16(np.around(circles))

    for i in circles[0, :]:
        cv2.circle(img, (i[0], i[1]), radius(img), (255, 0, 0), 2)

    return img, circles[0, :]


def find_unique_object(image_path: str) -> Tuple[int, int]:
    img, circles = annotate_objects(image_path)

    regions = []
    for x, y, r in circles:
        roi = img[y - r : y + r, x - r : x + r]
        hist = cv2.calcHist([roi], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        regions.append(hist)

    dist = cdist(regions, regions)
    i, j = np.unravel_index(np.argmax(dist), dist.shape)
    the_x, the_y = circles[i][:2]

    cv2.circle(img, (the_x, the_y), radius(img), (0, 255, 0), 3)

    return the_x, the_y
