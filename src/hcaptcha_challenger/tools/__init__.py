# -*- coding: utf-8 -*-
# Time       : 2023/8/19 17:52
# Author     : QIN2DIM
# GitHub     : https://github.com/QIN2DIM
# Description: Tools package for hCaptcha challenge solving.
"""
hCaptcha Challenger Tools
=========================

This module provides AI-powered tools for solving various hCaptcha challenge types.

Available Tools:
    - ImageClassifier: 9-grid image selection challenges
    - ChallengeRouter: Challenge type classification with prompt extraction
    - ChallengeClassifier: Alias for ChallengeRouter (backward compatibility)
    - SpatialPathReasoner: Drag-and-drop path challenges
    - SpatialPointReasoner: Point/area selection challenges
    - SpatialBboxReasoner: Bounding box challenges

Base Classes:
    - Reasoner: Abstract base class for all reasoning tools

Note:
    The `providers` module contains LLM provider implementations and can be
    imported separately for custom provider injection.
"""

# Base class for reasoning tools
from .base import Reasoner

# Challenge classification tools
from .challenge_router import ChallengeClassifier, ChallengeRouter

# Image classification tool
from .image_classifier import ImageClassifier

# Spatial reasoning tools
from .spatial import SpatialPathReasoner, SpatialPointReasoner, SpatialBboxReasoner

__all__ = [
    # Base class
    "Reasoner",
    # Challenge routing
    "ChallengeClassifier",
    "ChallengeRouter",
    # Image classification
    "ImageClassifier",
    # Spatial reasoning
    "SpatialPathReasoner",
    "SpatialPointReasoner",
    "SpatialBboxReasoner",
]
