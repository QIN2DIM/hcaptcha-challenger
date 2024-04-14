# -*- coding: utf-8 -*-
# Time       : 2024/4/14 12:25
# Author     : QIN2DIM
# GitHub     : https://github.com/QIN2DIM
# Description:
from fastapi import APIRouter

router = APIRouter()


@router.get("/sync", response_model=str)
async def datalake_sync_from_github_repo():
    pass


@router.get("/list", response_model=list)
async def datalake_list():
    pass
