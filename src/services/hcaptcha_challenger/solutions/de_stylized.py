# -*- coding: utf-8 -*-
# Time       : 2022/4/30 17:00
# Author     : Bingjie Yan
# Github     : https://github.com/beiyuouo
# Description:
from typing import Optional

from ._kernel import Solutions


class DeStylized(Solutions):
    def __init__(self, path_rainbow: Optional[str] = None):
        super(DeStylized, self).__init__(flag="de-stylized", path_rainbow=path_rainbow)

    def style_filter(self, img):
        """de-stylized"""

    def solution(self, img_stream, **kwargs) -> bool:
        """Implementation process of solution"""
        raise NotImplementedError


class ElephantDrawnWithLeaves(DeStylized):
    """Handle challenge 「Please select all the elephants drawn with lеaves」"""

    def solution(self, img_stream, **kwargs) -> bool:
        """Implementation process of solution"""


class HorsesDrawnWithFlowers(DeStylized):
    """Handle challenge「Please select all the horses drawn with flowers」"""

    def solution(self, img_stream, **kwargs) -> bool:
        """Implementation process of solution"""
