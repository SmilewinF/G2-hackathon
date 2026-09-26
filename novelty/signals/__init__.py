"""Scoring signals: the ``Signal`` interface and calibration helpers (``RobustScale``,
``normal_cdf``, ``smoothstep``) from ``base``, and the default implementations."""

from .base import Kind, RobustScale, Signal, SignalResult, normal_cdf, smoothstep
from .modifiers import ContentQuality, DuplicateCheck, Specificity, StanceRarity
from .novelty import ClauseCoverage, WholeTextNovelty
from ..errors import CalibrationError  # re-exported: callers (e.g. tests/test_math.py) import it from novelty.signals
from .relevance import TopicDiscriminant, TopicMargin

__all__ = [
    "CalibrationError",
    "ClauseCoverage",
    "ContentQuality",
    "DuplicateCheck",
    "Kind",
    "RobustScale",
    "Signal",
    "SignalResult",
    "Specificity",
    "StanceRarity",
    "TopicDiscriminant",
    "TopicMargin",
    "WholeTextNovelty",
    "normal_cdf",
    "smoothstep",
]
