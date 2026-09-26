"""The signal interface every scoring component implements.

How signals combine (see ``NoveltyScorer._combine``):

    novelty = min(NOVELTY signals) × Π(MODIFIER signals)
    gate    = Π(RELEVANCE signals)
    score   = novelty × gate

- NOVELTY: independent estimates of "is this new?"; taking the min means every view must agree.
- MODIFIER: multiplicative adjustments/vetoes on novelty (copy detection, low content, stance).
- RELEVANCE: gates that stop off-topic content earning anything regardless of novelty.

To add a component (an LLM relevance judge, a toxicity filter, ...): subclass ``Signal``, pick a
kind, and pass it in ``NoveltyScorer(signals=[...])`` or extend ``default_signals``.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from ..index import Analysis, ReferenceIndex

MIN_SCALE = 0.01  # floor for robust spread, so a near-constant corpus cannot blow up z-scores
MAD_TO_SIGMA = 1.4826  # makes the MAD a consistent estimate of the standard deviation for normal data


class Kind(str, Enum):
    NOVELTY = "novelty"
    MODIFIER = "modifier"
    RELEVANCE = "relevance"


@dataclass
class SignalResult:
    value: float  # always in [0, 1]
    reasons: list[str] = field(default_factory=list)
    detail: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not (0.0 <= self.value <= 1.0) or math.isnan(self.value):
            raise ValueError(f"signal value {self.value} outside [0, 1]")


class Signal(ABC):
    name: str
    kind: Kind

    def fit(self, index: ReferenceIndex) -> None:
        """Calibrate from scratch against the current reference index."""

    def update(self, index: ReferenceIndex, added: range) -> None:
        """Recalibrate after entries ``added`` were inserted. Defaults to a full ``fit``; signals
        with expensive calibration override it with an incremental update.

        This is also the initial calibration: the scorer calls ``update`` at construction with the
        whole seed corpus in ``added``. It calls ``fit`` only as a fallback when ``update`` raises
        an exception other than ``NoveltyError``. An override must therefore fall back to ``fit``
        while it holds no calibrated state (as ``WholeTextNovelty`` and ``ClauseCoverage`` do
        through ``_version``)."""
        self.fit(index)

    @abstractmethod
    def evaluate(self, analysis: Analysis, index: ReferenceIndex) -> SignalResult:
        """Assess one analysed submission against the index (read-only); the value is in [0, 1]."""


def normal_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def smoothstep(lo: float, hi: float, x: float) -> float:
    t = min(max((x - lo) / (hi - lo), 0.0), 1.0)
    return t * t * (3.0 - 2.0 * t)


@dataclass(frozen=True)
class RobustScale:
    """Median / MAD of a reference distribution; maps a raw value to [0, 1] via the normal CDF."""

    median: float
    scale: float

    @classmethod
    def fit(cls, values: np.ndarray) -> RobustScale:
        if len(values) == 0:
            raise ValueError("cannot calibrate on an empty distribution")
        med = float(np.median(values))
        mad = float(np.median(np.abs(values - med))) * MAD_TO_SIGMA
        return cls(med, max(mad, MIN_SCALE))

    def z(self, x: float) -> float:
        return (x - self.median) / self.scale

    def __call__(self, x: float) -> float:
        return normal_cdf(self.z(x))
