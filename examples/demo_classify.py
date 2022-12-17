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
    dir_datas = "horse_walking_or_running"
    images = [
        open(Path(__file__).parent.joinpath(f"{dir_datas}/{fn}"), "rb").read()
        for fn in os.listdir(Path(__file__).parent.joinpath(dir_datas))
    ]

    challenger = solver.new_challenger()
    if result := challenger.classify(prompt=prompt, images=images):
        for i, fp in enumerate(os.listdir(Path(__file__).parent.joinpath(dir_datas))):
            print(result[i], os.path.basename(fp))


if __name__ == "__main__":
    bytedance()
