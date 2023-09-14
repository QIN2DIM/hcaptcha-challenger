# -*- coding: utf-8 -*-
# Time       : 2023/9/14 13:19
# Author     : QIN2DIM
# GitHub     : https://github.com/QIN2DIM
# Description:
import pytest

from automation.sentinel import Sentinel
from hcaptcha_challenger.utils import SiteKey


@pytest.mark.parametrize("sitekey", [SiteKey.epic, SiteKey.discord, SiteKey.user_easy])
async def test_sentinel(sitekey: str):
    sentinel = Sentinel()
    sentinel.pending_sitekey.append(sitekey)
    await sentinel.bytedance()
