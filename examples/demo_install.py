# -*- coding: utf-8 -*-
# Time       : 2022/1/20 16:16
# Author     : QIN2DIM
# Github     : https://github.com/QIN2DIM
# Description:
from __future__ import annotations

import hcaptcha_challenger as solver

if __name__ == "__main__":
    solver.install(upgrade=False, pypi=False, flush_yolo=False, clip=False)
