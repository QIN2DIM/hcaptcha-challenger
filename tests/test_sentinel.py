# -*- coding: utf-8 -*-
# Time       : 2023/9/14 13:19
# Author     : QIN2DIM
# GitHub     : https://github.com/QIN2DIM
# Description:
import asyncio

from automation.sentinel import Sentinel


def test_sentinel():
    sentinel = Sentinel()
    asyncio.run(sentinel.bytedance())
