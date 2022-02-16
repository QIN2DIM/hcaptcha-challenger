# -*- coding: utf-8 -*-
# Time       : 2022/2/15 17:42
# Author     : QIN2DIM
# Github     : https://github.com/QIN2DIM
# Description:
import os
from os.path import join, dirname

from services.utils import ToolBox

HCAPTCHA_DEMO_SITES = [
    "https://maximedrn.github.io/hcaptcha-solver-python-selenium/"
]
# ---------------------------------------------------
# [√]工程根目录定位
# ---------------------------------------------------
# 系统根目录
PROJECT_ROOT = dirname(dirname(__file__))
# 文件数据库目录
PROJECT_DATABASE = join(PROJECT_ROOT, "database")
# YOLO模型
DIR_MODEL = join(PROJECT_ROOT, "model")
# 运行缓存目录
DIR_TEMP_CACHE = join(PROJECT_DATABASE, "temp_cache")
# 挑战缓存
DIR_CHALLENGE = join(DIR_TEMP_CACHE, "_challenge")
# 服务日志目录
DIR_LOG = join(PROJECT_DATABASE, "logs")
# ---------------------------------------------------
# [√]服务器日志配置
# ---------------------------------------------------
logger = ToolBox.init_log(
    error=join(DIR_LOG, "error.log"), runtime=join(DIR_LOG, "runtime.log")
)
# ---------------------------------------------------
# 路径补全
# ---------------------------------------------------
for _pending in [
    PROJECT_DATABASE,
    DIR_MODEL,
    DIR_TEMP_CACHE,
    DIR_CHALLENGE,
    DIR_LOG,
]:
    if not os.path.exists(_pending):
        os.mkdir(_pending)
