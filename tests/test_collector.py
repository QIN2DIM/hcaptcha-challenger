# -*- coding: utf-8 -*-
# Time       : 2023/9/14 13:14
# Author     : QIN2DIM
# GitHub     : https://github.com/QIN2DIM
# Description:

import pytest

from automation.collector import Collector

pytest_plugins = ("pytest_asyncio",)


@pytest.mark.asyncio
async def test_collector():
    collector = Collector()
    collector.prelude_tasks()
    await collector.startup_collector()
    collector.post_datasets()
