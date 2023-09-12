# -*- coding: utf-8 -*-
# Time       : 2022/8/5 8:45
# Author     : QIN2DIM
# Github     : https://github.com/QIN2DIM
# Description:
import asyncio
import os
import sys
from pathlib import Path

from istockphoto import Istock

phrase = "llama on a garden"
tmp_dir = Path(__file__).parent

if __name__ == "__main__":
    istock = Istock.from_phrase(phrase, tmp_dir)
    istock.pages = 4
    asyncio.run(istock.mining())

    if "win32" in sys.platform:
        os.startfile(istock.store_dir)
