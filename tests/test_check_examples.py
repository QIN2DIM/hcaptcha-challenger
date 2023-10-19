# -*- coding: utf-8 -*-
# Time       : 2023/10/19 18:38
# Author     : QIN2DIM
# GitHub     : https://github.com/QIN2DIM
# Description:


def test_common_classifier():
    from examples import demo_common_classifier

    demo_common_classifier.bytedance()


def test_common_detector():
    from examples import demo_common_detector

    demo_common_detector.bytedance()


def test_draw_animal_head():
    from examples import demo_draw_animal_head

    demo_draw_animal_head.demo()


def test_draw_segment_masks():
    from examples import demo_draw_animal_head

    demo_draw_animal_head.demo()


def test_find_unique_object():
    from examples import demo_find_unique_object

    demo_find_unique_object.demo()


def test_rank_largest_animal():
    from examples import demo_rank_largest_animal

    demo_rank_largest_animal.bytedance()
