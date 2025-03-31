# -*- coding: utf-8 -*-
# Time       : 2023/11/16 0:23
# Author     : QIN2DIM
# GitHub     : https://github.com/QIN2DIM
# Description:
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class CaptchaResponse(BaseModel):
    class Token(BaseModel):
        req: str
        type: str = "hsw"

    c: Token
    """
    type: hsw
    req: eyj0 ...
    """

    is_pass: bool | None = Field(default=False, alias="pass")
    """
    true or false
    """

    expiration: int | None = None
    """
    Return only when the challenge passes. (Optional)
    """

    generated_pass_UUID: str | None = ""
    """
    Return only when the challenge passes. (Optional)
    P1_eyj0 ...
    """

    error: str | None = ""
    """
    Return only when the challenge failure. (Optional)
    """


class RequestType(str, Enum):
    ImageLabelBinary = "image_label_binary"
    ImageLabelAreaSelect = "image_label_area_select"
    ImageLabelMultipleChoice = "image_label_multiple_choice"
