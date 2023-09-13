# -*- coding: utf-8 -*-
# Time       : 2022/8/5 8:45
# Author     : QIN2DIM
# Github     : https://github.com/QIN2DIM
# Description:
import asyncio
from pathlib import Path
from typing import Tuple

import pytest
from istockphoto import Istock

tmp_dir = Path(__file__).parent

phrases = ["airplane", "boat", "coffee"]
name2similar = [("hummingbird", "108176594")]


@pytest.mark.parametrize("phrase", phrases)
def test_select_phrase(phrase: str):
    istock = Istock.from_phrase(phrase, tmp_dir)
    istock.pages = 2
    asyncio.run(istock.mining())


@pytest.mark.parametrize("phrase_with_id", name2similar)
def test_similar_phrase(phrase_with_id: Tuple[str, str] | None):
    if not phrase_with_id:
        return

    phrase, istock_id = phrase_with_id
    istock = Istock.from_phrase(phrase, tmp_dir)
    istock.pages = 2
    istock.more_like_this(istock_id)
    asyncio.run(istock.mining())
