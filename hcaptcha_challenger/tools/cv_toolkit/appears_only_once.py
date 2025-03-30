# -*- coding: utf-8 -*-
# Time       : 2023/9/29 14:19
# Author     : QIN2DIM
# GitHub     : https://github.com/QIN2DIM
# Description:
from typing import Tuple, List, Literal

import cv2
import numpy as np
from skimage.metrics import structural_similarity as compare_ssim


def limited_radius(img) -> int:
    # x1, y1, x2, y2 = 90, 200, 413, 523
    r = 48 if 512 in img.shape else 32
    return r


def annotate_objects(image_path: str):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT, 1, 66, param1=100, param2=40, minRadius=5, maxRadius=65
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


def _build_mask(img: np.ndarray, circles: List[List[int]], lookup: Literal["color", "object"]):
    mask_images = []

    for circle in circles:
        x, y, r = circle
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        mask = cv2.circle(mask, (x, y), r, (255, 255, 0), -1)
        mask_img = cv2.bitwise_and(img, img, mask=mask)
        mask_img = mask_img[y - r : y + r, x - r : x + r]
        if lookup == "color":
            mask_img[np.where((mask_img == [0, 0, 0]).all(axis=2))] = [255, 255, 255]
        mask_images.append(mask_img)

    # uniform size, brightness, contrast, etc.
    max_size = max([mask_img.shape[0] for mask_img in mask_images])
    mask_images = [cv2.resize(mask_img, (max_size, max_size)) for mask_img in mask_images]

    return mask_images


def find_unique_object(img: np.ndarray, circles: List[List[int]]) -> Tuple[int, int, int]:
    mask_images = _build_mask(img, circles, lookup="object")

    similarity = []

    for i, mask_img in enumerate(mask_images):
        sig_sim = []
        for j, mask_img_ in enumerate(mask_images):
            if i == j:
                sig_sim.append(0)
                continue
            score, _ = compare_ssim(mask_img, mask_img_, win_size=3, full=True)
            sig_sim.append(score)
        similarity.append(np.array(sig_sim))

    similarity = np.array(similarity)
    sum_similarity = np.sum(similarity, axis=0)
    unique_index = np.argmin(sum_similarity)
    unique_circle = circles[unique_index]

    return unique_circle[0], unique_circle[1], unique_circle[2]


def find_unique_color(img: np.ndarray, circles: List[List[int]]) -> Tuple[int, int, int]:
    mask_images = _build_mask(img, circles, lookup="color")

    original_mask_images = mask_images.copy()
    original_mask_images = np.array(original_mask_images)

    mask_images = [cv2.convertScaleAbs(mask_img, alpha=1.5, beta=0) for mask_img in mask_images]

    scores = []

    for original_mask_img, mask_img in zip(original_mask_images, mask_images):
        gray = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)
        center_color = gray[gray.shape[0] // 2, gray.shape[1] // 2]
        gray[gray < min(200, int(center_color))] = 0

        # remove noise
        kernel = np.ones((3, 3), np.uint8)
        gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)

        # binary
        threshold = 255
        gray[gray < threshold] = 0
        gray[gray >= threshold] = 255

        colors = original_mask_img[gray == 0]
        colors = np.array(colors).reshape(-1, 3)
        score = np.var(colors, axis=0).sum()

        scores.append(score)

    unique_index = np.argmin(scores)

    unique_circle = circles[unique_index]

    return unique_circle[0], unique_circle[1], unique_circle[2]
