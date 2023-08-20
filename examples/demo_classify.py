# -*- coding: utf-8 -*-
# Time       : 2022/9/23 17:28
# Author     : QIN2DIM
# Github     : https://github.com/QIN2DIM
# Description:
import os
from pathlib import Path

import hcaptcha_challenger as solver

# Init local-side of the ModelHub
solver.install()

prompt = "Please click each image containing a horse walking or running."

label_dir = Path(__file__).parent.joinpath("horse_walking_or_running")

images = [label_dir.joinpath(fn).read_bytes() for fn in os.listdir(label_dir)]


def bytedance():
    classifier = solver.BinaryClassifier()
    if result := classifier.execute(prompt, images):
        for i, name in enumerate(os.listdir(label_dir)):
            print(result[i], name)


if __name__ == "__main__":
    bytedance()
