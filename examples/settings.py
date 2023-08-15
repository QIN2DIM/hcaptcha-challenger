# -*- coding: utf-8 -*-
# Time       : 2022/2/15 17:42
# Author     : QIN2DIM
# Github     : https://github.com/QIN2DIM
# Description:
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    motion_temp_dir: Path = Path(__file__).parent.joinpath("temp")

    HCAPTCHA_DEMO_API = "https://accounts.hcaptcha.com/demo?sitekey={}"
    SITE_KEYS = {
        "epic": "91e4137f-95af-4bc9-97af-cdcedce21c8c",
        "hcaptcha": "a5f74b19-9e45-40e0-b45d-47ff91b7a6c2",
        "discord": "f5561ba9-8f1e-40ca-9b5b-a0b3f719ef34",
        "oracle": "d857545c-9806-4f9e-8e9d-327f565aeb46",
        "publisher": "c86d730b-300a-444c-a8c5-5312e7a93628",
    }

    # https://www.wappalyzer.com/technologies/security/hcaptcha/
    HCAPTCHA_DEMO_SITES = [
        # [√] label: Tags follow point-in-time changes
        HCAPTCHA_DEMO_API.format(SITE_KEYS["publisher"]),
        # [√] label: `vertical river`
        HCAPTCHA_DEMO_API.format(SITE_KEYS["oracle"]),
        # [x] label: `airplane in the sky flying left`
        HCAPTCHA_DEMO_API.format(SITE_KEYS["discord"]),
        # [√] label: hcaptcha-challenger
        HCAPTCHA_DEMO_API.format(SITE_KEYS["hcaptcha"]),
    ]
