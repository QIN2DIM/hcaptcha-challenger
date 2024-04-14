# -*- coding: utf-8 -*-
# Time       : 2024/4/14 12:24
# Author     : QIN2DIM
# GitHub     : https://github.com/QIN2DIM
# Description:
from .challenge import router as challenge_router
from .datalake import router as datalake_router

__all__ = ["challenge_router", "datalake_router"]
