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

label_dir = Path(__file__).parent.joinpath("diamond_bracelet")

images = [label_dir.joinpath(fn).read_bytes() for fn in os.listdir(label_dir)]


def bytedance():
    classifier = solver.BinaryClassifier()
    if result := classifier.execute(prompt, images):
        for i, name in enumerate(os.listdir(label_dir)):
            print(f"{name} - {result[i]}")


if __name__ == "__main__":
    bytedance()
