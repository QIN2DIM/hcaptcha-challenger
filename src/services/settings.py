# -*- coding: utf-8 -*-
# Time       : 2022/2/15 17:42
# Author     : QIN2DIM
# Github     : https://github.com/QIN2DIM
# Description:
import os
from os.path import join, dirname

from services.utils import ToolBox

HCAPTCHA_DEMO_SITES = [
    "https://maximedrn.github.io/hcaptcha-solver-python-selenium/"
]
# ---------------------------------------------------
# [√]Lock the project directory
# ---------------------------------------------------
# Source root directory
PROJECT_ROOT = dirname(dirname(__file__))

# File database directory
PROJECT_DATABASE = join(PROJECT_ROOT, "database")

# The storage directory of the YOLO object detection model
DIR_MODEL = join(PROJECT_ROOT, "model")

# Run cache directory
DIR_TEMP_CACHE = join(PROJECT_DATABASE, "temp_cache")

# Directory for challenge images
DIR_CHALLENGE = join(DIR_TEMP_CACHE, "_challenge")

# Service log directory
DIR_LOG = join(PROJECT_DATABASE, "logs")
# ---------------------------------------------------
# [√]Server log configuration
# ---------------------------------------------------
logger = ToolBox.init_log(
    error=join(DIR_LOG, "error.log"), runtime=join(DIR_LOG, "runtime.log")
)
# ---------------------------------------------------
# [√]Path completion
# ---------------------------------------------------
for _pending in [
    PROJECT_DATABASE,
    DIR_MODEL,
    DIR_TEMP_CACHE,
    DIR_CHALLENGE,
    DIR_LOG,
]:
    if not os.path.exists(_pending):
        os.mkdir(_pending)
