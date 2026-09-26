"""Relevance signals: does the submission remain about the fixed content?"""

from __future__ import annotations

import numpy as np

from ..errors import CalibrationError
from ..index import Analysis, ReferenceIndex, unit
from .base import Kind, Signal, SignalResult, smoothstep


def _spd_solve(s: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Solve s x = v for a symmetric positive definite s: Cholesky, then forward and back
    substitution. For the 384 x 384 shrinkage matrix this took 3.8 ms against 8.2 ms for
    np.linalg.solve (whose multithreaded LU costs more than it saves at this size), same result."""
    chol = np.linalg.cholesky(s)
    n = len(v)
    y, x = np.empty(n), np.empty(n)
    for i in range(n):
        y[i] = (v[i] - chol[i, :i] @ y[:i]) / chol[i, i]
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - chol[i + 1 :, i] @ x[i + 1 :]) / chol[i, i]
    return x


def _equal_error_boundary(pos: np.ndarray, neg: np.ndarray) -> float:
    """The training projection t where the share of positives below t (missed on-topic comments)
    is closest to the share of negatives at or above it (let-through off-topic ones); the smallest
    such t on ties. Vectorised over the sorted candidates.

    The balanced-accuracy optimum used before is an argmax over an overlap region with two nearly
    equal peaks: one ordinary on-topic comment added to the corpus moved it from 19.2 to 28.2 and
    cut a novel probe's gate from 1.0 to 0.14. This crossing of two empirical CDFs moves by at most
    one data point per added comment."""
    candidates = np.unique(np.r_[pos, neg])
    fnr = np.searchsorted(np.sort(pos), candidates, side="left") / len(pos)
    fpr = 1.0 - np.searchsorted(np.sort(neg), candidates, side="left") / len(neg)
    return float(candidates[int(np.argmin(np.abs(fnr - fpr)))])


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
    loo: dict[str, float]  # leave-one-out margin of each on-topic corpus submission, set by fit()
    on_topic_margin: float  # median of loo; margins are divided by it

    def __init__(self, floor: float = 0.1, full: float = 0.5, min_on_topic_margin: float = 0.01) -> None:
        self.floor = floor
        self.full = full
        self.min_on_topic_margin = min_on_topic_margin

    def fit(self, index: ReferenceIndex) -> None:
        subs = index.submission_indices
        members = subs[index.on_topic_mask[subs]]
        if len(members) == 0:
            raise CalibrationError("no on-topic submissions to calibrate relevance against")
        # Leave-one-out margin of every member at once: v·unit(T − v) − v·g, using
        # ‖T − v‖² = ‖T‖² − 2 T·v + ‖v‖² instead of one centroid per member.
        topic = index.topic_sum().astype(np.float64)
        v = index.content[members].astype(np.float64)
        tv, vv = v @ topic, np.einsum("ij,ij->i", v, v)
        norms = np.sqrt(np.maximum(topic @ topic - 2 * tv + vv, 1e-24))
        margins = (tv - vv) / norms - v @ index.generic_vec
        self.loo = {index.entries[i].id: float(m) for i, m in zip(members, margins)}
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

    def clause_relevance(self, a: Analysis, index: ReferenceIndex) -> np.ndarray:
        """1 for a clause whose own margin is positive, else 0: this gate's per-clause test is
        unchanged from before the learned gate existed."""
        return (a.clause_margins > 0).astype(float)

    def corpus_relevance(self) -> dict[str, float]:
        return {k: round(self.relevance(m), 4) for k, m in self.loo.items()}


class TopicDiscriminant(Signal):
    """Relevance learned for THIS article from labelled examples.

    The contrastive margin above measures "closer to the topic than to generic chatter", so any
    same-town civic comment (bus cuts, library hours, water rates) passes: all such comments sit
    in one narrow band of embedding similarity. This signal instead fits a shrinkage Fisher
    discriminant: one direction in embedding space that separates the on-topic references (the
    article and every on-topic submission) from article-specific hard negatives (same-town
    comments about other subjects, data/relevance_negatives.json) plus the generic anchors.

    - Shrinkage (default 0.9) regularises the covariance toward the identity, which is what makes
      ~130 training vectors in 384 dimensions generalise; it was chosen on the evaluation set's
      dev split and was stable under resampling of the negatives.
    - The boundary is the equal-error projection on the training vectors (as many on-topic
      references below it as hard negatives above it); the gate rises from 0 to 1 across
      boundary +/- band * (median on-topic projection - boundary). The balanced-accuracy optimum
      used before gave the same boundary on the seed corpus but jumped when ordinary comments were
      admitted (see ``_equal_error_boundary``).
    - Scored on the substantive body, like TopicMargin, so headlines and keyword lists cannot buy
      relevance. With fewer than MIN_NEGATIVES hard negatives it falls back to TopicMargin.
    """

    name = "relevance"
    kind = Kind.RELEVANCE
    MIN_NEGATIVES = 10

    # Calibrated state, set by fit().
    w: np.ndarray
    boundary: float
    width: float

    def __init__(self, shrinkage: float = 0.9, band: float = 0.2, floor: float = 0.1, full: float = 0.5) -> None:
        self.shrinkage = shrinkage
        self.band = band
        self._fallback = TopicMargin(floor=floor, full=full)
        self._uses_fallback = False

    @property
    def uses_fallback(self) -> bool:
        return self._uses_fallback

    @staticmethod
    def _positives(index: ReferenceIndex) -> np.ndarray:
        return index.content[index.on_topic_mask].astype(np.float64)

    @staticmethod
    def _negatives(index: ReferenceIndex) -> np.ndarray:
        return np.vstack([index.negative_content, index.anchor_content]).astype(np.float64)

    def _direction(self, pos: np.ndarray, neg: np.ndarray, n_pos: int, sum_pos: np.ndarray,
                   outer_pos: np.ndarray, mean_neg: np.ndarray, scatter_neg: np.ndarray) -> np.ndarray:
        """Fisher direction w = S^-1 (mu_pos - mu_neg), S = (a/m) X'X + b I with a = 1 - s and
        b = s tr(X'X) / (m d): shrinkage s of the pooled within-class covariance of the m training
        rows X (d columns, rows centred per class) toward the identity.

        X'X comes from sufficient statistics, the positives' count, sum and sum of outer products
        plus the fixed negatives' scatter: outer_pos - n_pos mu_pos mu_pos' + scatter_neg. Adding a
        comment is then a rank-one update, not a pass over the corpus. The solve takes the cheaper
        of two exact forms: with fewer rows than dimensions (the seed corpus: m ~ 133, d = 384) the
        Woodbury identity (v - X' ((m b / a) I + X X')^-1 X v) / b, an m x m solve; beyond that a
        d x d one. Woodbury alone grew with the corpus: 40 ms per add at 1,000 entries."""
        mean_pos = sum_pos / n_pos
        scatter = outer_pos - n_pos * np.outer(mean_pos, mean_pos) + scatter_neg
        m, d = n_pos + len(neg), len(mean_pos)
        a = 1.0 - self.shrinkage
        b = self.shrinkage * float(np.trace(scatter)) / (m * d)  # s * tr(C) / d
        diff = mean_pos - mean_neg
        if a <= 0.0:
            return diff / b
        if m < d:
            x = np.vstack([pos - mean_pos, neg - mean_neg])
            return (diff - x.T @ np.linalg.solve((m * b / a) * np.eye(m) + x @ x.T, x @ diff)) / b
        return _spd_solve((a / m) * scatter + b * np.eye(d), diff)

    @staticmethod
    def _threshold(ps: np.ndarray, ns: np.ndarray) -> tuple[float, float]:
        boundary = _equal_error_boundary(ps, ns)
        return boundary, max(float(np.median(ps)) - boundary, 1e-6)

    def _train(self, pos: np.ndarray, neg: np.ndarray) -> tuple[np.ndarray, float, float]:
        """Direction, boundary and width trained from scratch on these rows."""
        mean_neg = neg.mean(axis=0)
        w = self._direction(pos, neg, len(pos), pos.sum(axis=0), pos.T @ pos, mean_neg,
                            (neg - mean_neg).T @ (neg - mean_neg))
        return w, *self._threshold(pos @ w, neg @ w)

    def fit(self, index: ReferenceIndex) -> None:
        self._uses_fallback = len(index.negative_content) < self.MIN_NEGATIVES
        if self._uses_fallback:
            self._fallback.fit(index)
            return
        pos, neg = self._positives(index), self._negatives(index)
        self._n_pos, self._sum_pos, self._outer_pos = len(pos), pos.sum(axis=0), pos.T @ pos
        self._mean_neg = neg.mean(axis=0)
        self._scatter_neg = (neg - self._mean_neg).T @ (neg - self._mean_neg)
        self._entries = len(index)
        self._refresh(index, pos, neg)

    def update(self, index: ReferenceIndex, added: range) -> None:
        if self._uses_fallback or getattr(self, "_entries", None) != added.start:
            return self.fit(index)
        new = [i for i in added if index.on_topic_mask[i]]
        rows = index.content[new].astype(np.float64)
        self._n_pos += len(rows)
        self._sum_pos = self._sum_pos + rows.sum(axis=0)
        self._outer_pos = self._outer_pos + rows.T @ rows
        self._entries = len(index)
        self._refresh(index, self._positives(index), self._negatives(index))

    def _refresh(self, index: ReferenceIndex, pos: np.ndarray, neg: np.ndarray) -> None:
        self._index = index
        self.w = self._direction(pos, neg, self._n_pos, self._sum_pos, self._outer_pos, self._mean_neg,
                                 self._scatter_neg)
        self.boundary, self.width = self._threshold(pos @ self.w, neg @ self.w)

    def _relevance(self, projection: float, boundary: float, width: float) -> float:
        lo, hi = boundary - self.band * width, boundary + self.band * width
        return min(max((projection - lo) / (hi - lo), 0.0), 1.0)

    def evaluate(self, a: Analysis, index: ReferenceIndex) -> SignalResult:
        if self._uses_fallback:
            return self._fallback.evaluate(a, index)
        if a.content_vec is None:
            return SignalResult(0.0, ["no substantive body to assess relevance"], {"relevance": 0.0, "margin": None})
        projection = float(a.content_vec.astype(np.float64) @ self.w)
        rel = self._relevance(projection, self.boundary, self.width)
        gate = rel * rel * (3.0 - 2.0 * rel)  # smoothstep across the band
        margin = projection - self.boundary
        reasons = []
        if gate == 0.0:
            reasons.append(f"relevance {rel:.2f} (margin {margin:+.3f}) is below floor: not rewarded")
        elif gate < 1.0:
            reasons.append(f"relevance {rel:.2f} partially gates the reward ({gate:.2f})")
        return SignalResult(gate, reasons, {"relevance": rel, "margin": margin})

    def clause_relevance(self, a: Analysis, index: ReferenceIndex) -> np.ndarray:
        """Relevance of each clause on its own, on the same learned direction and scale as the gate."""
        if self._uses_fallback:
            return self._fallback.clause_relevance(a, index)
        projections = a.clause_vecs.astype(np.float64) @ self.w
        return np.array([self._relevance(float(p), self.boundary, self.width) for p in projections])

    def corpus_relevance(self) -> dict[str, float]:
        """Leave-one-out relevance of each on-topic submission (retrains without it)."""
        if self._uses_fallback:
            return self._fallback.corpus_relevance()
        index = self._index
        rows = np.flatnonzero(index.on_topic_mask)
        members = [i for i in index.submission_indices if index.on_topic_mask[i]]
        pos_all, neg = self._positives(index), self._negatives(index)
        out = {}
        for i in members:
            w, boundary, width = self._train(pos_all[rows != i], neg)
            projection = float(index.content[i].astype(np.float64) @ w)
            out[index.entries[i].id] = round(self._relevance(projection, boundary, width), 4)
        return out
