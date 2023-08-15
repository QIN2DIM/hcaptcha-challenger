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


def bytedance():
    # Challenge prompt in Chinese or English
    prompt = "Please click each image containing a horse walking or running."

    # Absolute path to the Challenge Images
    label_name = "horse_walking_or_running"
    label_dir = Path(__file__).parent.joinpath(label_name)
    images = [label_dir.joinpath(fn).read_bytes() for fn in os.listdir(label_dir)]

    challenger = solver.new_challenger()
    if result := challenger.classify(prompt=prompt, images=images):
        for i, name in enumerate(os.listdir(label_dir)):
            print(result[i], name)


if __name__ == "__main__":
    bytedance()
