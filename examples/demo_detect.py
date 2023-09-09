# -*- coding: utf-8 -*-
# Time       : 2023/8/31 3:12
# Author     : QIN2DIM
# GitHub     : https://github.com/QIN2DIM
# Description:
import os
from pathlib import Path

import hcaptcha_challenger as solver

# Init local-side of the ModelHub
solver.install(flush_yolo=True)

# please click on the elephant
# please click on the raccoon
# please click on the digit 9
prompt = "please click on the elephant"

label_dir = Path(__file__).parent.joinpath("image_label_area_select", "raccoon")

images = [label_dir.joinpath(fn).read_bytes() for fn in os.listdir(label_dir)]


def bytedance():
    tool = solver.AreaSelector()
    results = tool.execute(prompt, images, shape_type="point")
    if not results:
        return
    for i, filename in enumerate(os.listdir(label_dir)):
        alts = results[i]
        if len(alts) > 1:
            alts = sorted(alts, key=lambda x: x[-1])
        if len(alts) > 0:
            best = alts[-1]
            class_name, (center_x, center_y), score = best
            print(f"{label_dir.name}.{filename} - {class_name=} {(center_x, center_y)=}")
        else:
            print(f"{label_dir.name}.{filename} - ObjectsNotFound")


if __name__ == "__main__":
    bytedance()
