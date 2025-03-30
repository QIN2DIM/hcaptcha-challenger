# -*- coding: utf-8 -*-
# Time       : 2023/10/19 18:38
# Author     : QIN2DIM
# GitHub     : https://github.com/QIN2DIM
# Description:


def test_demo_auto_labeling():
    import demo_auto_labeling

    demo_auto_labeling.run(startfile=False)


def test_demo_classifier_common():
    import demo_classifier_common

    demo_classifier_common.bytedance()


def test_rank_classifier_largest_animal():
    import demo_classifier_rank_largest_animal

    demo_classifier_rank_largest_animal.demo()


def test_demo_classifier_self_supervised():
    import demo_classifier_self_supervised

    demo_classifier_self_supervised.demo()


def test_demo_detector_common():
    import demo_detector_common

    demo_detector_common.bytedance()


def test_draw_segment_masks():
    import demo_draw_segment_masks

    demo_draw_segment_masks.demo(startfile=False)


def test_demo_draw_star_bricks():
    import demo_draw_star_bricks

    demo_draw_star_bricks.demo(startfile=False)


def test_find_unique_object():
    import demo_find_unique_object

    demo_find_unique_object.demo(startfile=False)
