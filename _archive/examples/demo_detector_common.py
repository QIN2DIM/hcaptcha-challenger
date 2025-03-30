# -*- coding: utf-8 -*-
# Time       : 2023/8/31 3:12
# Author     : QIN2DIM
# GitHub     : https://github.com/QIN2DIM
# Description:
import os
from pathlib import Path

import hcaptcha_challenger as solver

# Init local-side of the ModelHub
solver.install(upgrade=True)

prompt = "please click on the elephant"

assets_dir = Path(__file__).parent.parent.joinpath("assets")
images_dir = assets_dir.joinpath("image_label_area_select", "raccoon")

images = [images_dir.joinpath(fn).read_bytes() for fn in os.listdir(images_dir)]


def bytedance():
    tool = solver.AreaSelector()
    results = tool.execute(prompt, images, shape_type="point")
    if not results:
        return
    for i, filename in enumerate(os.listdir(images_dir)):
        alts = results[i]
        if len(alts) > 1:
            alts = sorted(alts, key=lambda x: x[-1])
        if len(alts) > 0:
            best = alts[-1]
            class_name, (center_x, center_y), score = best
            print(f"{images_dir.name}.{filename} - {class_name=} {(center_x, center_y)=}")
        else:
            print(f"{images_dir.name}.{filename} - ObjectsNotFound")


if __name__ == "__main__":
    bytedance()
