# -*- coding: utf-8 -*-
# Time       : 2023/10/16 4:56
# Author     : QIN2DIM
# GitHub     : https://github.com/QIN2DIM
# Description:
import os
from pathlib import Path

import hcaptcha_challenger as solver

solver.install(upgrade=True)

prompt = "Please click on the largest animal."

data_dir = Path(__file__).parent.joinpath("largest_animal")

pending_keys = ["hedgehog", "rabbit", "raccoon"]


def bytedance():
    for pk in pending_keys:
        example_paths = [data_dir.joinpath(f"example_{pk}.png")]
        images_dir = data_dir.joinpath(pk)
        image_paths = [images_dir.joinpath(image_name) for image_name in os.listdir(images_dir)]

        classifier = solver.BinaryClassifier()

        if result := classifier.execute(prompt, image_paths, example_paths):
            print(f">> Match model - resnet={classifier.model_name} prompt={prompt}")
            for i, image_path in enumerate(image_paths):
                desc = f"{image_path.parent.name}/{image_path.name}"
                print(f">> {desc} - result={result[i]}")


if __name__ == "__main__":
    bytedance()
