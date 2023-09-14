# -*- coding: utf-8 -*-
# Time       : 2023/9/14 13:14
# Author     : QIN2DIM
# GitHub     : https://github.com/QIN2DIM
# Description:
import asyncio

from automation.collector import Collector


def test_collector():
    collector = Collector()
    asyncio.run(collector.bytedance())
