"""Exceptions the novelty package raises on purpose.

Every error is a ``NoveltyError``, so callers can catch the package's failures in one place.
Each also subclasses the built-in it replaces (``ValueError`` / ``RuntimeError``), so existing
``except ValueError`` code keeps working.

    ValidationError   bad user input (a submission field, a fixed content over 100 words)
    DataError         a data file is missing or malformed (message names the file and item)
    CalibrationError  the reference data cannot support a meaningful calibration
    EmbeddingError    the embedding backend failed (model load, network, malformed vectors)
    ScoringError      a signal failed while scoring or recalibrating (message names the signal)
"""

from __future__ import annotations


class NoveltyError(Exception):
    """Base class for every error the novelty package raises deliberately."""


class ValidationError(NoveltyError, ValueError):
    pass


class DataError(NoveltyError, ValueError):
    pass


class CalibrationError(NoveltyError, ValueError):
    pass


class EmbeddingError(NoveltyError, RuntimeError):
    pass


class ScoringError(NoveltyError, RuntimeError):
    pass
