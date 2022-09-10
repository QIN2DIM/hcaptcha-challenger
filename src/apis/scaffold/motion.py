# -*- coding: utf-8 -*-
# Time       : 2022/7/20 5:46
# Author     : QIN2DIM
# Github     : https://github.com/QIN2DIM
# Description:
from services.motion_data.offload import MotionData
from services.settings import DIR_DATABASE


def train_motion(test_site: str):
    with MotionData(DIR_DATABASE) as motion:
        motion.mimic(test_site)
