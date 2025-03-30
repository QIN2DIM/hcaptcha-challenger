# -*- coding: utf-8 -*-
# Time       : 2022/2/15 17:43
# Author     : QIN2DIM
# GitHub     : https://github.com/QIN2DIM
# Description:
from __future__ import annotations

from pathlib import Path

from hcaptcha_challenger.models import QuestionResp, CaptchaResponse
from hcaptcha_challenger.utils import init_log

__all__ = ["QuestionResp", "CaptchaResponse"]

LOG_DIR = Path(__file__).parent.joinpath("logs", "{time:YYYY-MM-DD}")

init_log(
    runtime=LOG_DIR.joinpath("runtime.log"),
    error=LOG_DIR.joinpath("error.log"),
    serialize=LOG_DIR.joinpath("serialize.log"),
)
