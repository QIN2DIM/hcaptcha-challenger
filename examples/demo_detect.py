# -*- coding: utf-8 -*-
# Time       : 2023/8/31 3:12
# Author     : QIN2DIM
# GitHub     : https://github.com/QIN2DIM
# Description:
import os
from pathlib import Path

import hcaptcha_challenger as solver
from hcaptcha_challenger.components.image_label_area_select import AreaSelector

# Init local-side of the ModelHub
solver.install(flush_yolo=True)

prompt = "please click on the elephant"

label_dir = Path(__file__).parent.joinpath("animal")

images = [label_dir.joinpath(fn).read_bytes() for fn in os.listdir(label_dir)]


def bytedance():
    alts = []
    tool = AreaSelector()
    results = tool.execute(prompt, images, shape_type="point")
    for i, filename in enumerate(os.listdir(label_dir)):
        # name, (center_x, center_y), score = results[i]
        # alt = {"name": name, "position": {"x": center_x, "y": center_y}, "score": score}
        # alts.append(alt)
        print(f"{label_dir.name}.{filename} - {results[i]}")


if __name__ == "__main__":
    bytedance()
