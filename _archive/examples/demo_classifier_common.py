# -*- coding: utf-8 -*-
# Time       : 2022/9/23 17:28
# Author     : QIN2DIM
# Github     : https://github.com/QIN2DIM
# Description:
import os
from pathlib import Path
from typing import List

import hcaptcha_challenger as solver

# Init local-side of the ModelHub
solver.install(upgrade=True)

prompt = "diamond bracelet"

assets_dir = Path(__file__).parent.parent.joinpath("assets")
images_dir = assets_dir.joinpath("image_label_binary", "diamond_bracelet")


def get_test_images() -> List[Path]:
    image_paths = []
    for image_name in os.listdir(images_dir):
        image_path = images_dir.joinpath(image_name)
        if image_path.is_file():
            image_paths.append(image_path)

    return image_paths


def bytedance():
    image_paths = get_test_images()

    classifier = solver.BinaryClassifier()
    if results := classifier.execute(prompt, image_paths):
        for image_path, result in zip(image_paths, results):
            print(f"{image_path.name=} - {result=} {classifier.model_name=}")


if __name__ == "__main__":
    bytedance()
