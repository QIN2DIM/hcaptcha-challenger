# -*- coding: utf-8 -*-
# Time       : 2023/9/14 13:14
# Author     : QIN2DIM
# GitHub     : https://github.com/QIN2DIM
# Description:
import os

from automation.collector import Collector


async def test_collector():
    if not os.getenv("GITHUB_TOKEN"):
        return

    collector = Collector()
    collector.prelude_tasks()
    await collector.startup_collector()
    collector.post_datasets()
