# -*- coding: utf-8 -*-
# Time       : 2022/7/20 4:45
# Author     : QIN2DIM
# Github     : https://github.com/QIN2DIM
# Description:
import os.path

from sanic import Sanic, Request
from sanic.response import html

app = Sanic("motion-data")


@app.route("/")
async def test(request: Request):
    print(request.headers)
    fp = os.path.join(os.path.dirname(__file__), "motion.html")
    with open(fp, "rb") as file:
        return html(file.read())
