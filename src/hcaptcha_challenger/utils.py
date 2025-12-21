# -*- coding: utf-8 -*-
# Time       : 2023/8/19 17:19
# Author     : QIN2DIM
# GitHub     : https://github.com/QIN2DIM
# Description:
from __future__ import annotations

import os
import random
import string
import sys
import uuid
from pathlib import Path
from typing import Literal

import pytz
from loguru import logger


def init_log(**sink_channel):
    """
    Initialize the log configuration

    Parameter:
        sink_channel: A dictionary containing different log output channels
        - error: The path to the error log file
        - runtime: The path to the runtime log file
        - serialize: serialize the log file path
    """
    log_level = os.getenv("LOG_LEVEL", "DEBUG").upper()

    shanghai_tz = pytz.timezone("Asia/Shanghai")

    persistent_format = (
        "<g>{time:YYYY-MM-DD HH:mm:ss}</g> | "
        "<lvl>{level}</lvl>    | "
        "<c><u>{name}</u></c>:{function}:{line} | "
        "{message} - "
        "{extra}"
    )

    stdout_format = (
        "<g>{time:YYYY-MM-DD HH:mm:ss}</g> | "
        "<lvl>{level:<8}</lvl>    | "
        "<c>{name}</c>:<c>{function}</c>:<c>{line}</c> | "
        "<n>{message}</n>"
    )

    logger.remove()

    logger.add(
        sink=sys.stdout,
        colorize=True,
        level=log_level,
        format=stdout_format,
        diagnose=False,
        filter=lambda record: record["time"].replace(tzinfo=pytz.UTC).astimezone(shanghai_tz),
    )

    if error_sink := sink_channel.get("error"):
        logger.add(
            sink=error_sink,
            level="ERROR",
            rotation="5 MB",
            retention="7 days",
            encoding="utf8",
            diagnose=False,
            filter=lambda record: record["time"].replace(tzinfo=pytz.UTC).astimezone(shanghai_tz),
        )

    if runtime_sink := sink_channel.get("runtime"):
        logger.add(
            sink=runtime_sink,
            level="TRACE",
            rotation="5 MB",
            retention="7 days",
            encoding="utf8",
            diagnose=False,
            filter=lambda record: record["time"].replace(tzinfo=pytz.UTC).astimezone(shanghai_tz),
        )

    if serialize_sink := sink_channel.get("serialize"):
        logger.add(
            sink=serialize_sink,
            level="DEBUG",
            format=persistent_format,
            encoding="utf8",
            diagnose=False,
            serialize=True,
            filter=lambda record: record["time"].replace(tzinfo=pytz.UTC).astimezone(shanghai_tz),
        )

    return logger


class SiteKey:
    discord = "4c672d35-0701-42b2-88c3-78380b0db560"
    epic = "91e4137f-95af-4bc9-97af-cdcedce21c8c"
    hcaptcha = "a5f74b19-9e45-40e0-b45d-47ff91b7a6c2"
    hcaptcha_signup = "13257c82-e129-4f09-a733-2a7cb3102832"
    new_type_challenge = "ace50dd0-0d68-44ff-931a-63b670c7eed7"
    cloud_horse = "edc4ce89-8903-4906-80b1-7440ad9a69c8"
    top_level = "adafb813-8b5c-473f-9de3-485b4ad5aa09"

    user_easy = "c86d730b-300a-444c-a8c5-5312e7a93628"
    user = user_easy
    user_moderate = "eb932362-438e-43b4-9373-141064402110"
    user_difficult = "3fac610f-4879-4fd5-919b-ca072a134a79"

    @staticmethod
    def as_site_link(
        site_key: Literal["discord", "epic", "easy", "moderate", "difficult", "user"] | str,
    ):
        keymap = {
            "discord": SiteKey.discord,
            "epic": SiteKey.epic,
            "user": SiteKey.user_easy,
            "easy": SiteKey.user_easy,
            "moderate": SiteKey.user_moderate,
            "difficult": SiteKey.user_difficult,
        }
        url = "https://accounts.hcaptcha.com/demo"
        if site_key in keymap:
            return f"{url}?sitekey={keymap[site_key]}"

        try:
            uuid.UUID(site_key)
            return f"{url}?sitekey={site_key}"
        except ValueError:
            raise ValueError(f"sitekey is a string in UUID format, but you entered `{site_key}`")

    @staticmethod
    def choice():
        ks = [
            "f5561ba9-8f1e-40ca-9b5b-a0b3f719ef34",
            "91e4137f-95af-4bc9-97af-cdcedce21c8c",
            "a5f74b19-9e45-40e0-b45d-47ff91b7a6c2",
            "13257c82-e129-4f09-a733-2a7cb3102832",
            "ace50dd0-0d68-44ff-931a-63b670c7eed7",
            "c86d730b-300a-444c-a8c5-5312e7a93628",
            "edc4ce89-8903-4906-80b1-7440ad9a69c8",
            "adafb813-8b5c-473f-9de3-485b4ad5aa09",
        ]
        k = random.choice(ks)
        return f"https://accounts.hcaptcha.com/demo?sitekey={k}"


def load_desc(path: Path, substitutions: dict[str, str] | None = None) -> str:
    """Load a tool description from a file, with optional substitutions."""
    description = path.read_text(encoding="utf-8")
    if substitutions:
        description = string.Template(description).safe_substitute(substitutions)
    if description and isinstance(description, str):
        description = description.strip()
    return description
