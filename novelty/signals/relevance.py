"""Relevance signals: does the submission remain about the fixed content?"""

from __future__ import annotations

import numpy as np

from ..index import Analysis, ReferenceIndex, unit
from .base import Kind, Signal, SignalResult, smoothstep


class CalibrationError(ValueError):
    """The reference data cannot support a meaningful calibration."""


class TopicMargin(Signal):
    """Contrastive relevance of the submission's *substantive body*.

    margin = sim(content, topic centroid) − sim(content, generic-chatter centroid). A margin ≤ 0
    means the text reads more like unrelated comments than like a response to this article.

    Measured on the substantive body clauses only (not the headline, not keyword lists), so
    appending "garage park Elm Street levy" to unrelated text, or stuffing the headline, does not
    buy relevance. Normalised by the typical corpus member's leave-one-out margin, then gated.
    """

    name = "relevance"
    kind = Kind.RELEVANCE

    def __init__(self, floor: float = 0.1, full: float = 0.5, min_on_topic_margin: float = 0.01) -> None:
        self.floor = floor
        self.full = full
        self.min_on_topic_margin = min_on_topic_margin

    def fit(self, index: ReferenceIndex) -> None:
        topic = index.topic_sum()
        self.loo = {}
        for i in index.submission_indices:
            e = index.entries[i]
            if e.on_topic:
                v = index.content_vec(i)
                self.loo[e.id] = float(index.margin(v[None, :], topic - v)[0])
        if not self.loo:
            raise CalibrationError("no on-topic submissions to calibrate relevance against")
        self.on_topic_margin = float(np.median(list(self.loo.values())))
        if self.on_topic_margin < self.min_on_topic_margin:
            raise CalibrationError(
                f"typical corpus margin {self.on_topic_margin:.4f} is too small: the off-topic "
                "anchors are not distinguishable from the corpus with this embedder"
            )
        self._topic = unit(topic)

    def relevance(self, margin: float) -> float:
        return min(max(margin / self.on_topic_margin, 0.0), 1.0)

    def evaluate(self, a: Analysis, index: ReferenceIndex) -> SignalResult:
        if a.content_vec is None:
            return SignalResult(0.0, ["no substantive body to assess relevance"], {"relevance": 0.0, "margin": None})
        margin = float(a.content_vec @ self._topic - a.content_vec @ index.generic_vec)
        rel = self.relevance(margin)
        gate = smoothstep(self.floor, self.full, rel)
        reasons = []
        if gate == 0.0:
            reasons.append(f"relevance {rel:.2f} (margin {margin:+.3f}) is below floor: not rewarded")
        elif gate < 1.0:
            reasons.append(f"relevance {rel:.2f} partially gates the reward ({gate:.2f})")
        return SignalResult(gate, reasons, {"relevance": rel, "margin": margin})

    def corpus_relevance(self) -> dict[str, float]:
        return {k: round(self.relevance(m), 4) for k, m in self.loo.items()}
