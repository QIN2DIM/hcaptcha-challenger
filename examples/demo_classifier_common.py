# -*- coding: utf-8 -*-
# Time       : 2022/9/23 17:28
# Author     : QIN2DIM
# Github     : https://github.com/QIN2DIM
# Description:
import os
from pathlib import Path

import hcaptcha_challenger as solver

# Init local-side of the ModelHub
solver.install(upgrade=True)

prompt = "diamond bracelet"

assets_dir = Path(__file__).parent.parent.joinpath("assets")
images_dir = assets_dir.joinpath("image_label_binary", "diamond_bracelet")

images = [images_dir.joinpath(fn).read_bytes() for fn in os.listdir(images_dir)]


def bytedance():
    classifier = solver.BinaryClassifier()
    if result := classifier.execute(prompt, images):
        for i, name in enumerate(os.listdir(images_dir)):
            print(f"{name} - {result[i]}")


if __name__ == "__main__":
    bytedance()
