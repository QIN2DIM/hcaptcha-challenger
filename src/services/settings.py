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
    "hcaptcha": "a5f74b19-9e45-40e0-b45d-47ff91b7a6c2",
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
# hcaptcha-challenger
#  ├── database
#  │   ├── logs
#  │   ├── motion_data
#  │   └── temp_cache
#  │       ├── _challenge
#  │       └── captcha_screenshot
#  ├── model
#  │   ├── _assets
#  │   ├── _memory
#  │   └── rainbow.yaml[DEPRECATED]
#  └── src
#      ├── main.py
#      └── objects.yaml
# ---------------------------------------------------
PROJECT_SRC = dirname(dirname(__file__))

DIR_DATABASE = join(dirname(PROJECT_SRC), "database")
DIR_LOG = join(DIR_DATABASE, "logs")
DIR_TEMP_CACHE = join(DIR_DATABASE, "temp_cache")
DIR_CHALLENGE = join(DIR_TEMP_CACHE, "_challenge")

DIR_MODEL = join(dirname(PROJECT_SRC), "model")
DIR_ASSETS = join(DIR_MODEL, "_assets")

PATH_OBJECTS_YAML = join(PROJECT_SRC, "objects.yaml")

# ---------------------------------------------------
# [√]Server log configuration
# ---------------------------------------------------
logger = ToolBox.init_log(error=join(DIR_LOG, "error.log"), runtime=join(DIR_LOG, "runtime.log"))

# ---------------------------------------------------
# [√]Path completion
# ---------------------------------------------------
for _pending in (DIR_CHALLENGE, DIR_ASSETS):
    os.makedirs(_pending, exist_ok=True)
