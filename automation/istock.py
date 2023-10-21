# -*- coding: utf-8 -*-
# Time       : 2022/8/5 8:45
# Author     : QIN2DIM
# Github     : https://github.com/QIN2DIM
# Description:
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Tuple

from istockphoto import Istock

tmp_dir = Path(__file__).parent

phrases = ["cat"]
name2similar = [("hummingbird", "108176594")]


def select_phrase(phrase: str):
    istock = Istock.from_phrase(phrase, tmp_dir)
    istock.pages = 4
    asyncio.run(istock.mining())


def similar_phrase(phrase_with_id: Tuple[str, str] | None):
    if not phrase_with_id:
        return

    phrase, istock_id = phrase_with_id
    istock = Istock.from_phrase(phrase, tmp_dir)
    istock.pages = 2
    istock.more_like_this(istock_id)
    asyncio.run(istock.mining())


if __name__ == "__main__":
    select_phrase("elephant")
