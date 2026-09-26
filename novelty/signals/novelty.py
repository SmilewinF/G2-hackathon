"""Novelty signals: is this submission saying something the corpus has not already said?"""

from __future__ import annotations

import numpy as np

from ..errors import CalibrationError
from ..index import Analysis, ReferenceIndex
from .base import Kind, RobustScale, Signal, SignalResult


class WholeTextNovelty(Signal):
    """Distance of the whole submission to its nearest neighbours, relative to the corpus.

    raw = w × (1 − nearest similarity) + (1 − w) × (1 − mean top-k similarity), on hybrid
    similarity; calibrated against every corpus item's leave-one-out raw value (median / MAD →
    normal CDF). 0.5 = as novel as a typical existing submission.

    Calibration keeps each entry's top-k neighbour similarities. Inserting entry j only needs
    j's similarity row (O(n)) and a vectorised top-k merge for the rows j beats; a full rebuild
    happens only when the index's lexical vectors were refit.
    """

    name = "whole_text"
    kind = Kind.NOVELTY
    Z_REASON_PREFIX = "novelty z="  # start of the reason every evaluation adds; NoveltyScorer's result log line omits it
    # Calibrated state, set by fit()/update(). Annotations only, so hasattr() is False until then.
    loo: dict[str, float]  # leave-one-out raw novelty of each corpus submission, by id
    scale: RobustScale  # calibration of raw novelty

    def __init__(self, k: int = 5, nearest_weight: float = 0.5) -> None:
        self.k = k
        self.nearest_weight = nearest_weight
        self._version: int | None = None

    def _raw_top(self, top: np.ndarray) -> np.ndarray:
        w = self.nearest_weight
        return w * (1.0 - top[..., 0]) + (1.0 - w) * (1.0 - top.mean(axis=-1))

    def _top(self, row: np.ndarray) -> np.ndarray:
        """The k largest values of ``row``, in descending order."""
        k = self.k
        if len(row) <= k:
            return np.sort(row)[::-1]
        return np.sort(row[np.argpartition(row, -k)[-k:]])[::-1]

    def raw(self, sims: np.ndarray) -> float:
        """Uncalibrated novelty of one similarity row (reported as ``raw_novelty``)."""
        return float(self._raw_top(self._top(sims)))

    def fit(self, index: ReferenceIndex) -> None:
        if len(index.submission_indices) <= self.k:
            raise CalibrationError(f"corpus needs more than k={self.k} submissions to calibrate novelty")
        self._topk = np.full((len(index), self.k), -np.inf)
        for i in range(len(index)):
            row = index.entry_sims(i)
            row[i] = -np.inf
            self._topk[i] = self._top(row)
        self._version = index.version
        self._finish(index)

    def update(self, index: ReferenceIndex, added: range) -> None:
        if self._version != index.version:
            return self.fit(index)
        self._topk = np.vstack([self._topk, np.full((len(added), self.k), -np.inf)])
        for j in added:
            s = index.entry_sims(j)[:j]  # similarity to entries inserted before j
            self._topk[j] = self._top(s)
            old = self._topk[:j]
            beats = s > old[:, -1]
            if beats.any():
                merged = np.concatenate([old[beats], s[beats, None]], axis=1)
                merged.sort(axis=1)
                old[beats] = merged[:, ::-1][:, : self.k]
        self._finish(index)

    def _finish(self, index: ReferenceIndex) -> None:
        subs = index.submission_indices
        raw = self._raw_top(self._topk[subs])
        self.loo = {index.entries[i].id: float(r) for i, r in zip(subs, raw)}
        self.scale = RobustScale.fit(raw)

    def evaluate(self, a: Analysis, index: ReferenceIndex) -> SignalResult:
        raw = self.raw(a.sims)
        z = self.scale.z(raw)
        nearest = int(np.argmax(a.sims))
        return SignalResult(
            self.scale(raw),
            [f"{self.Z_REASON_PREFIX}{z:+.2f} vs. corpus (nearest {index.entries[nearest].id} @ {a.sims[nearest]:.2f})"],
            {"raw": raw, "z": z},
        )


class ClauseCoverage(Signal):
    """Is at least one *relevant, substantive* clause new?

    Whole-text similarity misses two attacks: a "kitchen sink" comment that restates many
    existing takes in one text (it sits between clusters, so it looks far from each one), and a
    stock take padded with unrelated text. Scoring each clause against the corpus and keeping the
    most novel on-topic one fixes both: every clause of a kitchen-sink comment is already covered,
    and padding is off-topic so it cannot supply the novelty.

    It applies only when the body has at least two substantive clauses. With exactly one, that
    clause is the whole text, so the signal stays neutral (1.0, ``applied=False``) and leaves it to
    ``WholeTextNovelty``; with none, it returns 0.

    Calibration keeps each stored clause's best similarity to any *other* entry; inserting entry
    j is one clause-column (O(clauses)) plus rows for j's own clauses.
    """

    name = "clause_coverage"
    kind = Kind.NOVELTY
    scale: RobustScale  # calibration of 1 - best clause similarity, set by fit()/update()

    def __init__(self) -> None:
        self._version: int | None = None

    def _clause_best(self, index: ReferenceIndex, c: int) -> float:
        s = index.clause_sims(c)
        s[index.clause_owner[c]] = -np.inf  # leave-one-out: a clause is trivially covered by its own comment
        return float(s.max())

    def _in_calibration_set(self, index: ReferenceIndex, c: int) -> bool:
        """Clause ``c`` counts toward calibration: substantive, and owned by a submission (not the article)."""
        return index.clause_substantive[c] and index.entries[index.clause_owner[c]].is_submission

    def fit(self, index: ReferenceIndex) -> None:
        n = len(index.clause_owner)
        self._best = np.full(n, -np.inf)
        self._mask = np.array([self._in_calibration_set(index, c) for c in range(n)], dtype=bool)
        for c in np.flatnonzero(self._mask):
            self._best[c] = self._clause_best(index, c)
        self._version = index.version
        self._finish(index)

    def update(self, index: ReferenceIndex, added: range) -> None:
        if self._version != index.version:
            return self.fit(index)
        before = len(self._best)
        total = len(index.clause_owner)
        self._best = np.concatenate([self._best, np.full(total - before, -np.inf)])
        new_mask = np.array([self._in_calibration_set(index, c) for c in range(before, total)], dtype=bool)
        self._mask = np.concatenate([self._mask, new_mask])
        for j in added:  # existing clauses gain one more entry they may be covered by
            col = index.clause_sims_to_entry(j)[:before]
            np.maximum(self._best[:before], col, out=self._best[:before])
        for c in before + np.flatnonzero(new_mask):  # new clauses: best match among all other entries
            self._best[c] = self._clause_best(index, c)
        self._finish(index)

    def _finish(self, index: ReferenceIndex) -> None:
        self.scale = RobustScale.fit(1.0 - self._best[self._mask])

    def evaluate(self, a: Analysis, index: ReferenceIndex) -> SignalResult:
        substantive = np.array(a.substantive, dtype=bool)
        if not substantive.any():
            return SignalResult(0.0, ["no substantive clause to assess"], {"clause": None, "applied": True})
        if substantive.sum() < 2:
            # One clause *is* the whole text, which WholeTextNovelty already scores. Clause-vs-
            # comment similarity is unreliable for a lone short clause (a concise new idea
            # shares its topic words with longer comments), so stay neutral in the min().
            return SignalResult(1.0, [], {"clause": None, "applied": False})
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
        return SignalResult(values[best], reasons, {"clause": clause, "applied": True})
