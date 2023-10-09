# -*- coding: utf-8 -*-
# Time       : 2023/9/29 14:19
# Author     : QIN2DIM
# GitHub     : https://github.com/QIN2DIM
# Description:
from typing import Tuple, List

import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import rbf_kernel


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


def find_unique_object(img: np.ndarray, circles: List[List[int]]) -> Tuple[int, int, int]:
    def spectral_clustering():
        num_clusters = 2

        S = rbf_kernel(feature_vectors)
        D = np.diag(np.sum(S, axis=1))
        L = D - S

        eigvals, eigvecs = np.linalg.eigh(L)
        eigvecs = eigvecs[:, :num_clusters]

        kmeans = KMeans(n_clusters=num_clusters, n_init=10)
        kmeans.fit(eigvecs)

        return kmeans.labels_

    if len(circles) == 1:
        x, y, r = circles[0]
        return x, y, r

    feature_vectors = []
    for x, y, r in circles:
        roi = img[y - r : y + r, x - r : x + r]
        hist = cv2.calcHist([roi], [0, 1, 2], None, [16, 16, 16], [0, 256, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        feature_vectors.append(hist)

    labels = spectral_clustering()

    unique, counts = np.unique(labels, return_counts=True)
    car_1_label = unique[np.argmin(counts)]

    for i, label in enumerate(labels):
        if label == car_1_label:
            x, y, r = circles[i][0:3]
            return x, y, r
