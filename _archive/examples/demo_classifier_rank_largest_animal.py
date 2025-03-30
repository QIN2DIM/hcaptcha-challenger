# -*- coding: utf-8 -*-
# Time       : 2023/10/16 4:56
# Author     : QIN2DIM
# GitHub     : https://github.com/QIN2DIM
# Description:
import os
from pathlib import Path
from typing import List

import hcaptcha_challenger as solver

solver.install(upgrade=True)

prompt = "Please click on the largest animal."

assets_dir = Path(__file__).parent.parent.joinpath("assets")

pending_keys = ["hedgehog", "rabbit", "raccoon"]


def bytedance(image_paths: List[Path], example_paths: List[Path] = None):
    classifier = solver.BinaryClassifier()
    if results := classifier.execute(prompt, image_paths, example_paths):
        for image_path, result in zip(image_paths, results):
            desc = f"{image_path.parent.name}/{image_path.name}"
            print(f">> {desc=} - {result=} {classifier.model_name=}")


def demo():
    for pk in pending_keys:
        images_dir = assets_dir.joinpath("please click on the largest animal")
        example_paths = [images_dir.joinpath(f"example_{pk}.png")]
        images_dir = images_dir.joinpath(pk)
        image_paths = [images_dir.joinpath(image_name) for image_name in os.listdir(images_dir)]

        # Regular interface
        bytedance(image_paths, example_paths)


if __name__ == "__main__":
    demo()
