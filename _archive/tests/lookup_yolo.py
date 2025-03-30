# -*- coding: utf-8 -*-
# Time       : 2023/9/18 21:10
# Author     : QIN2DIM
# GitHub     : https://github.com/QIN2DIM
# Description:
from hcaptcha_challenger.onnx.modelhub import ModelHub

ash_samples = [
    "please click on the head of the animal catonacoach",
    "please click on the head of the animal default",  # Fixme
    "please click on the head of the animal foxonthesnow",
    "please click on the head of the animal goatonthesand",
    "please click on the head of the animal horseonagrassfield",
    "please click on the head of the animal hummingbirdonaflowerfield",
    "please click on the number two digit2",
    "please click on the squirrel squirrelwatercolor-lmv2",
    "please click on the squirrel squirrelwatercolorlmv2",
    "please click on the thumbnail of the animal that does not belong to the sea default",  # fixme
    "please click on the thumbnail of the animal that does not belong to the sea fantasia-lm-catwatercolor-sea",
    "please click on the thumbnail of the animal that does not belong to the sea fantasia-lm-elephantwatercolor-sea",
    "please click on the thumbnail of the animal that is unique fantasia-lm-sharkunderwater-sea",
    "please click on the thumbnail of the animal that is unique fantasia-lm-starfishunderwater-sea",
    "please click on the thumbnail that is not a fruit notfruit-raccoonwatercolor-lm",
    "please click on the thumbnail that is not an animal default",
    "please click on the thumbnail that is not an animal notanimalbananaonbasket-lm",
    "please click on the treasure default",
    "please click on the treasure lm-treasurechest",
]
modelhub = ModelHub.from_github_repo()
modelhub.parse_objects()

print(modelhub.ashes_of_war)


def lookup_yolo(ash: str):
    if "default" in ash:
        for focus_name, classes in modelhub.lookup_ash_of_war(ash):
            print(f"<-- {focus_name=} {ash=} ")

    else:
        focus_name, classes = modelhub.apply_ash_of_war(ash)
        print(f"<-- {focus_name=} {ash=} ")

    input("Continue >> ")


for s in ash_samples:
    lookup_yolo(s)
