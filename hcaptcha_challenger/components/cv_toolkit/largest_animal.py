# -*- coding: utf-8 -*-
# Time       : 2023/10/9 16:18
# Author     : QIN2DIM
# GitHub     : https://github.com/QIN2DIM
# Description:
from pathlib import Path
from typing import List

import cv2
from sklearn.cluster import SpectralClustering


def get_2d_image(path: Path):
    image = cv2.imread(str(path))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray


def extract_features(img):
    hog = cv2.HOGDescriptor()
    h = hog.compute(img)
    return h


def find_similar_objects(example_paths: List[Path], challenge_paths: List[Path]):
    example_num = len(example_paths)

    results: List[bool | None] = [None] * len(challenge_paths)

    images_example = [get_2d_image(path) for path in example_paths]
    example_shape = images_example[0].shape[0]

    images_challenge = []
    for i, path in enumerate(challenge_paths):
        image = get_2d_image(path)
        if image.shape[0] != example_shape:
            results[i] = False
        images_challenge.append(image)

    images_merged = images_example + images_challenge

    X = [extract_features(img) for img in images_merged]

    clf = SpectralClustering(n_clusters=2, affinity="nearest_neighbors", n_neighbors=5)
    y = clf.fit_predict(X)

    ref_img_idx = 0
    ref_label = y[ref_img_idx]

    sim_idx_sequence = [i for i, rl in enumerate(y) if rl == ref_label]

    for idx in sim_idx_sequence[example_num:]:
        fit_idx = idx - example_num
        if results[fit_idx] is None:
            results[fit_idx] = True

    return results
