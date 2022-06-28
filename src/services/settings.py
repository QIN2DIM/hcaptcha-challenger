# -*- coding: utf-8 -*-
# Time       : 2022/2/15 17:42
# Author     : QIN2DIM
# Github     : https://github.com/QIN2DIM
# Description:
import os
from os.path import join, dirname

from services.utils import ToolBox

HCAPTCHA_DEMO_API = "https://accounts.hcaptcha.com/demo?sitekey={}"
_SITE_KEYS = {
    "epic": "91e4137f-95af-4bc9-97af-cdcedce21c8c",
    "hcaptcha": "00000000-0000-0000-0000-000000000000",
    "discord": "f5561ba9-8f1e-40ca-9b5b-a0b3f719ef34",
    "oracle": "d857545c-9806-4f9e-8e9d-327f565aeb46",
    "publisher": "c86d730b-300a-444c-a8c5-5312e7a93628",
}

# https://www.wappalyzer.com/technologies/security/hcaptcha/
HCAPTCHA_DEMO_SITES = [
    # [√] label: Tags follow point-in-time changes
    HCAPTCHA_DEMO_API.format(_SITE_KEYS["publisher"]),
    # [√] label: `vertical river`
    HCAPTCHA_DEMO_API.format(_SITE_KEYS["oracle"]),
    # [x] label: `airplane in the sky flying left`
    HCAPTCHA_DEMO_API.format(_SITE_KEYS["discord"]),
    # [√] label: hcaptcha-challenger
    HCAPTCHA_DEMO_API.format(_SITE_KEYS["hcaptcha"]),
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

PATH_RAINBOW = join(DIR_MODEL, "rainbow.yaml")

# Run cache directory
DIR_TEMP_CACHE = join(PROJECT_DATABASE, "temp_cache")

# Directory for challenge images
DIR_CHALLENGE = join(DIR_TEMP_CACHE, "_challenge")

# Service log directory
DIR_LOG = join(PROJECT_DATABASE, "logs")
# ---------------------------------------------------
# [√]Server log configuration
# ---------------------------------------------------
logger = ToolBox.init_log(error=join(DIR_LOG, "error.log"), runtime=join(DIR_LOG, "runtime.log"))
# ---------------------------------------------------
# [√]Path completion
# ---------------------------------------------------
for _pending in [PROJECT_DATABASE, DIR_MODEL, DIR_TEMP_CACHE, DIR_CHALLENGE, DIR_LOG]:
    if not os.path.exists(_pending):
        os.mkdir(_pending)
