import os
from pathlib import Path

import cv2
from tqdm import tqdm

import hcaptcha_challenger as solver
from hcaptcha_challenger.onnx.modelhub import ModelHub
from hcaptcha_challenger.onnx.yolo import YOLOv8Seg

solver.install(upgrade=True)

this_dir = Path(__file__).parent
examples_dir = this_dir.parent.joinpath("examples")
figs_dir = examples_dir.joinpath("figs-seg")
figs_out_dir = this_dir.joinpath("figs-seg-out")


def test_yolov8seg():
    figs_out_dir.mkdir(exist_ok=True)

    modelhub = ModelHub.from_github_repo()
    modelhub.parse_objects()

    model_name = "appears_only_once_2309_yolov8s-seg.onnx"
    classes = modelhub.ashes_of_war.get(model_name)
    session = modelhub.match_net(model_name)
    yoloseg = YOLOv8Seg.from_pluggable_model(session, classes)

    pending_image_paths = [figs_dir.joinpath(image_name) for image_name in os.listdir(figs_dir)]

    total = len(pending_image_paths)
    desc_in = f'"{figs_dir.parent.name}/{figs_dir.name}"'
    with tqdm(total=total, desc=f"Labeling | {desc_in}") as progress:
        for image_path in pending_image_paths:
            img = cv2.imread(str(image_path))

            yoloseg(img)

            combined_img = yoloseg.draw_masks(img)
            output_path = figs_out_dir.joinpath(image_path.name)
            cv2.imwrite(str(output_path), combined_img)

            progress.update(1)


def test_yolov8seg_ash():
    modelhub = ModelHub.from_github_repo()
    modelhub.parse_objects()

    ash = "please click on the object that appears only once"
    for focus_name, classes in modelhub.lookup_ash_of_war(ash):
        print(focus_name, classes)
