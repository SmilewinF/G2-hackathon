"""Novelty signals: is this submission saying something the corpus has not already said?"""

from __future__ import annotations

import numpy as np

from ..index import Analysis, ReferenceIndex
from .base import Kind, RobustScale, Signal, SignalResult


class WholeTextNovelty(Signal):
    """Distance of the whole submission to its nearest neighbours, relative to the corpus.

    raw = w × (1 − nearest similarity) + (1 − w) × (1 − mean top-k similarity), on hybrid
    similarity; calibrated against every corpus item's leave-one-out raw value (median / MAD →
    normal CDF). 0.5 = as novel as a typical existing submission.
    """

    name = "whole_text"
    kind = Kind.NOVELTY

    def __init__(self, k: int = 5, nearest_weight: float = 0.5) -> None:
        self.k = k
        self.nearest_weight = nearest_weight

    def raw(self, sims: np.ndarray) -> float:
        top = np.sort(sims)[::-1][: self.k]
        w = self.nearest_weight
        return w * (1.0 - float(top[0])) + (1.0 - w) * (1.0 - float(np.mean(top)))

    def fit(self, index: ReferenceIndex) -> None:
        subs = index.submission_indices
        if len(subs) <= self.k:
            raise ValueError(f"corpus needs more than k={self.k} submissions")
        sims = index.pairwise()
        np.fill_diagonal(sims, -np.inf)
        self.loo = {index.entries[i].id: self.raw(sims[i]) for i in subs}
        self.scale = RobustScale.fit(np.array(list(self.loo.values())))

    def evaluate(self, a: Analysis, index: ReferenceIndex) -> SignalResult:
        raw = self.raw(a.sims)
        z = self.scale.z(raw)
        nearest = int(np.argmax(a.sims))
        return SignalResult(
            self.scale(raw),
            [f"novelty z={z:+.2f} vs. corpus (nearest {index.entries[nearest].id} @ {a.sims[nearest]:.2f})"],
            {"raw": raw, "z": z},
        )


class ClauseCoverage(Signal):
    """Is at least one *relevant, substantive* clause new?

    Whole-text similarity misses two attacks: a "kitchen sink" comment that restates many
    existing takes in one text (it sits between clusters, so it looks far from each one), and a
    stock take padded with unrelated text. Scoring each clause against the corpus and keeping the
    most novel on-topic one fixes both: every clause of a kitchen-sink comment is already covered,
    and padding is off-topic so it cannot supply the novelty.
    """

    name = "clause_coverage"
    kind = Kind.NOVELTY

    def fit(self, index: ReferenceIndex) -> None:
        raws = []
        for i in index.submission_indices:
            e = index.entries[i]
            if not e.clauses:
                continue
            sims = index.clause_sims_of(i)
            sims[:, i] = -np.inf  # leave-one-out: a clause is trivially covered by its own comment
            raws.extend(1.0 - sims.max(axis=1)[np.array(e.substantive)])
        self.scale = RobustScale.fit(np.array(raws))

    def evaluate(self, a: Analysis, index: ReferenceIndex) -> SignalResult:
        substantive = np.array(a.substantive, dtype=bool)
        if not substantive.any():
            return SignalResult(0.0, ["no substantive clause to assess"], {"clause": None})
        on_topic = substantive & (a.clause_margins > 0)
        # With no on-topic clause, fall back to all substantive ones: novelty stays a pure
        # "is it new" measure and the relevance gate is what zeroes off-topic content.
        eligible = np.flatnonzero(on_topic if on_topic.any() else substantive)
        values = [self.scale(1.0 - float(a.clause_sims[j].max())) for j in eligible]
        best = int(np.argmax(values))
        clause = a.clauses[eligible[best]]
        reasons = []
        if len(eligible) > 1 and values[best] < 0.5:
            reasons.append("every on-topic clause is already covered by existing submissions")
        return SignalResult(values[best], reasons, {"clause": clause})
