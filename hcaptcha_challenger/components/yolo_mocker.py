# -*- coding: utf-8 -*-
# Time       : 2023/10/14 19:14
# Author     : QIN2DIM
# GitHub     : https://github.com/QIN2DIM
# Description:
from __future__ import annotations

import os
import shutil
from pathlib import Path

import cv2
import onnxruntime
from tqdm import tqdm

from hcaptcha_challenger.onnx.modelhub import ModelHub
from hcaptcha_challenger.onnx.modelhub import request_resource
from hcaptcha_challenger.onnx.yolo import YOLOv8


class CcYOLO:
    model_url = "https://github.com/QIN2DIM/hcaptcha-challenger/releases/download/model/"

    def __init__(self, model_name: str, images_absolute_dir: Path, this_dir: Path):
        assert images_absolute_dir

        input_dir = images_absolute_dir
        if not input_dir.absolute():
            raise ValueError("`images_absolute_dir` Should be an absolute path")
        if not input_dir.is_dir():
            raise FileNotFoundError("`images_absolute_dir` pointed to does not exist")

        self.modelhub = ModelHub.from_github_repo()
        self.modelhub.parse_objects()

        self.model_name = model_name
        self.input_dir = input_dir

        self.output_dir = this_dir
        if input_dir.parent.name:
            self.output_dir = self.output_dir.joinpath(input_dir.parent.name)
        self.output_dir = self.output_dir.joinpath(input_dir.name)
        self.output_miss_dir = self.output_dir.joinpath("miss")

        self.output_miss_dir.mkdir(parents=True, exist_ok=True)

    @property
    def desc_out(self):
        return f'"{self.output_dir.parent.name}/{self.output_dir.name}"'

    @property
    def desc_in(self):
        return f'"{self.input_dir.parent.name}/{self.input_dir.name}"'

    def get_model(self) -> YOLOv8 | None:
        classes = self.modelhub.ashes_of_war.get(self.model_name)
        if not classes:
            raise AttributeError(f"Model name not found - {self.model_name=}")

        print(f">> Match model - {self.model_name=}")
        model_path = self.modelhub.models_dir.joinpath(self.model_name)
        if not model_path.exists():
            request_resource(self.model_url + self.model_name, model_path)

        session = onnxruntime.InferenceSession(
            model_path, providers=onnxruntime.get_available_providers()
        )
        detector = YOLOv8.from_pluggable_model(session, classes)

        return detector

    def spawn(self):
        """
        ```python
        import os
        import sys
        from pathlib import Path

        from hcaptcha_challenger.components.yolo_mocker import CcYOLO


        def run():
            model_name = "head_of_the_animal_2310_yolov8s.onnx"
            images_dir = "image_label_area_select/please click on the head of the animal/default"

            this_dir = Path(__file__).parent
            output_dir = this_dir.joinpath("yolo_mocker")
            images_dir = this_dir.joinpath(images_dir).absolute()

            ccy = CcYOLO(model_name, images_dir, output_dir)
            ccy.spawn()

            if "win32" in sys.platform:
                os.startfile(ccy.output_dir)


        if __name__ == "__main__":
            run()
        ```
        :return:
        """

        def draw():
            alts = sorted(results, key=lambda x: x[-1])
            text, ps, pe, _ = alts[-1]
            image = cv2.imread(str(image_path))
            pt1 = int(ps[0]), int(ps[1])
            pt2 = int(pe[0]), int(pe[1])
            cv2.rectangle(image, pt1, pt2, (87, 241, 126), 2)
            cv2.imwrite(str(output_path), image)

        if not (detector := self.get_model()):
            return

        pending_image_paths = []
        for image_name in os.listdir(self.input_dir):
            image_path = self.input_dir.joinpath(image_name)
            if image_path.is_file() and not self.output_dir.joinpath(image_name).exists():
                pending_image_paths.append(image_path)

        total = len(pending_image_paths)
        handle, miss = 0, 0

        with tqdm(total=total, desc=f"Labeling | {self.desc_in}") as progress:
            for image_path in pending_image_paths:
                results = detector(image_path, shape_type="bounding_box")
                progress.update(1)
                if not results:
                    output_miss_path = self.output_miss_dir.joinpath(image_path.name)
                    shutil.copyfile(image_path, output_miss_path)
                    miss += 1
                    continue
                output_path = self.output_dir.joinpath(image_path.name)
                draw()
                handle += 1
        print(f">> Statistic - {total=} {handle=} {miss=} - input --> {self.desc_in}")
