# -*- coding: utf-8 -*-
# Time       : 2023/9/7 0:58
# Author     : QIN2DIM
# GitHub     : https://github.com/QIN2DIM
# Description:
import os
from pathlib import Path

import hcaptcha_challenger as solver
from hcaptcha_challenger.components.image_label_area_select import AreaSelector
from hcaptcha_challenger.onnx.yolo import is_matched_ash_of_war

# Init local-side of the ModelHub
solver.install(flush_yolo=True)

cleaning_prompt = "draw a tight bounding box around the car"

label_dir = Path(__file__).parent.joinpath(
    "image_label_area_select", "draw a tight bounding box around the car"
)

images = [label_dir.joinpath(fn).read_bytes() for fn in os.listdir(label_dir)]


def bytedance():
    tool = AreaSelector()
    results = tool.execute(cleaning_prompt, images, shape_type="bounding_box")
    if not results:
        return

    alts = []
    for i, filename in enumerate(os.listdir(label_dir)):
        # alts[0] like ('car', (0.6850128, 181.35565), (266.95172, 441.93317), 0.89994013)
        for name, (x1, y1), (x2, y2), score in results[i]:
            if not is_matched_ash_of_war(ash=cleaning_prompt, class_name=name):
                continue
            scoop = (x2 - x1) * (y2 - y1)
            start = (int(x1), int(y1))
            end = (int(x2), int(y2))
            alt = {"name": name, "start": start, "end": end, "scoop": scoop}
            alts.append(alt)

        if len(alts) > 1:
            alts = sorted(alts, key=lambda xf: xf["scoop"])
        if len(alts) > 0:
            best = alts[-1]
            x1, y1 = best["start"]
            x2, y2 = best["end"]
            # await locator.click(delay=200, position={"x": x1, "y": y1})
            # await self.page.mouse.move(x2, y2)
            # await locator.click(delay=200, position={"x": x2, "y": y2})
        else:
            print(f"{label_dir.name}.{filename} - ObjectsNotFound")


if __name__ == "__main__":
    bytedance()
