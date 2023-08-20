# -*- coding: utf-8 -*-
# Time       : 2023/8/19 17:17
# Author     : QIN2DIM
# GitHub     : https://github.com/QIN2DIM
# Description:
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

from playwright.sync_api import BrowserContext as SyncContext

from hcaptcha_challenger.agents.skeleton import Skeleton


@dataclass
class PlaywrightAgent(Skeleton):
    def switch_to_challenge_frame(self, ctx, **kwargs):
        pass

    def get_label(self, ctx, **kwargs):
        pass

    def mark_samples(self, ctx, *args, **kwargs):
        pass

    def challenge(self, ctx, model, *args, **kwargs):
        pass

    def is_success(self, ctx, *args, **kwargs) -> Tuple[str, str]:
        pass

    def anti_checkbox(self, ctx, *args, **kwargs):
        pass

    def anti_hcaptcha(self, ctx, *args, **kwargs) -> bool | str:
        pass


def apply_stealth(context: SyncContext):
    enabled_evasions = [
        "chrome.app",
        "chrome.csi",
        "chrome.loadTimes",
        "chrome.runtime",
        "iframe.contentWindow",
        "media.codecs",
        "navigator.hardwareConcurrency",
        "navigator.languages",
        "navigator.permissions",
        "navigator.plugins",
        "navigator.webdriver",
        "sourceurl",
        "webgl.vendor",
        "window.outerdimensions",
    ]

    for e in enabled_evasions:
        evasion_code = (
            Path(__file__)
            .parent.joinpath(f"puppeteer-extra-plugin-stealth/evasions/{e}/index.js")
            .read_text(encoding="utf8")
        )
        context.add_init_script(evasion_code)

    return context
