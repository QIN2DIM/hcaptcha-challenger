# -*- coding: utf-8 -*-
# Time       : 2023/8/19 17:19
# Author     : QIN2DIM
# GitHub     : https://github.com/QIN2DIM
# Description:
from __future__ import annotations

import inspect
import sys
from typing import Dict, Any, Literal

from loguru import logger


def init_log(**sink_channel):
    event_logger_format = "<g>{time:YYYY-MM-DD HH:mm:ss}</g> | <lvl>{level}</lvl> - {message}"
    persistent_format = (
        "<g>{time:YYYY-MM-DD HH:mm:ss}</g> | "
        "<lvl>{level}</lvl>    | "
        "<c><u>{name}</u></c>:{function}:{line} | "
        "{message} - "
        "{extra}"
    )
    serialize_format = event_logger_format + " - {extra}"
    logger.remove()
    logger.add(
        sink=sys.stdout, colorize=True, level="DEBUG", format=serialize_format, diagnose=False
    )
    if sink_channel.get("error"):
        logger.add(
            sink=sink_channel.get("error"),
            level="ERROR",
            rotation="1 week",
            encoding="utf8",
            diagnose=False,
            format=persistent_format,
        )
    if sink_channel.get("runtime"):
        logger.add(
            sink=sink_channel.get("runtime"),
            level="DEBUG",
            rotation="20 MB",
            retention="20 days",
            encoding="utf8",
            diagnose=False,
            format=persistent_format,
        )
    if sink_channel.get("serialize"):
        logger.add(
            sink=sink_channel.get("serialize"),
            level="DEBUG",
            format=persistent_format,
            encoding="utf8",
            diagnose=False,
            serialize=True,
        )
    return logger


def from_dict_to_model(cls, data: Dict[str, Any]):
    return cls(
        **{
            key: (data[key] if val.default == val.empty else data.get(key, val.default))
            for key, val in inspect.signature(cls).parameters.items()
        }
    )


class SiteKey:
    discord = "f5561ba9-8f1e-40ca-9b5b-a0b3f719ef34"
    epic = "91e4137f-95af-4bc9-97af-cdcedce21c8c"
    hcaptcha = "a5f74b19-9e45-40e0-b45d-47ff91b7a6c2"
    hcaptcha_signup = "13257c82-e129-4f09-a733-2a7cb3102832"
    new_type_challenge = "ace50dd0-0d68-44ff-931a-63b670c7eed7"
    user = "c86d730b-300a-444c-a8c5-5312e7a93628"
    cloud_horse = "edc4ce89-8903-4906-80b1-7440ad9a69c8"
    top_level = "adafb813-8b5c-473f-9de3-485b4ad5aa09"

    @staticmethod
    def to_sitelink() -> str:
        return f"https://accounts.hcaptcha.com/demo?sitekey={SiteKey.user}"

    @staticmethod
    def as_sitelink(sitekey: Literal["discord", "epic", "hcaptcha", "newtype", "user"] | str):
        keymap = {
            "discord": SiteKey.discord,
            "epic": SiteKey.epic,
            "hcaptcha": SiteKey.hcaptcha,
            "newtype": SiteKey.new_type_challenge,
            "user": SiteKey.user,
        }
        return f"https://accounts.hcaptcha.com/demo?sitekey={keymap.get(sitekey, sitekey)}"
