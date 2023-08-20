# -*- coding: utf-8 -*-
# Time       : 2022/1/20 16:16
# Author     : QIN2DIM
# Github     : https://github.com/QIN2DIM
# Description:
from __future__ import annotations

import hcaptcha_challenger as solver


def do(upgrade: bool | None = False, username="QIN2DIM", lang="en"):
    """Download the dependencies required to run the project."""
    solver.install(upgrade=upgrade, username=username, lang=lang)


if __name__ == "__main__":
    do(upgrade=True)
