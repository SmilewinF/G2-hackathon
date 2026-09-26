"""Relevance signals: does the submission remain about the fixed content?"""

from __future__ import annotations

import numpy as np

from ..errors import CalibrationError
from ..index import Analysis, ReferenceIndex, unit
from .base import Kind, Signal, SignalResult, smoothstep


def _balanced_boundary(pos: np.ndarray, neg: np.ndarray) -> float:
    """The training projection t that maximises balanced accuracy, mean(pos >= t) and
    mean(neg < t) averaged; the smallest such t on ties. Vectorised over the sorted candidates."""
    candidates = np.unique(np.r_[pos, neg])
    pos_sorted, neg_sorted = np.sort(pos), np.sort(neg)
    tpr = (len(pos) - np.searchsorted(pos_sorted, candidates, side="left")) / len(pos)
    tnr = np.searchsorted(neg_sorted, candidates, side="left") / len(neg)
    return float(candidates[int(np.argmax(tpr + tnr))])


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
    - The boundary is the projection that maximises balanced accuracy on the training vectors;
      the gate rises from 0 to 1 across boundary +/- band * (median on-topic projection - boundary).
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

    def __init__(self, shrinkage: float = 0.9, band: float = 0.25, floor: float = 0.1, full: float = 0.5) -> None:
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

    def _train(self, pos: np.ndarray, neg: np.ndarray) -> tuple[np.ndarray, float, float]:
        """Fisher direction w = S^-1 (mu_pos - mu_neg), S = (1 - s) C + s (tr C / d) I, with C the
        pooled within-class covariance of the m training rows X (m x d, rows centred per class).

        C has rank <= m (~130) while d = 384, so w is computed with the Woodbury identity
        (a/m X'X + b I)^-1 v = (v - X' ((m b / a) I + X X')^-1 X v) / b, an m x m solve instead of
        d x d; the result is the same vector (up to float rounding)."""
        mu_p, mu_n = pos.mean(axis=0), neg.mean(axis=0)
        X = np.vstack([pos - mu_p, neg - mu_n])
        m, d = X.shape
        a = 1.0 - self.shrinkage
        b = self.shrinkage * float(np.einsum("ij,ij->", X, X)) / (m * d)  # s * tr(C) / d
        diff = mu_p - mu_n
        if a <= 0.0:
            w = diff / b
        else:
            inner = (m * b / a) * np.eye(m) + X @ X.T
            w = (diff - X.T @ np.linalg.solve(inner, X @ diff)) / b
        ps, ns = pos @ w, neg @ w
        boundary = _balanced_boundary(ps, ns)
        width = max(float(np.median(ps)) - boundary, 1e-6)
        return w, boundary, width

    def fit(self, index: ReferenceIndex) -> None:
        self._uses_fallback = len(index.negative_content) < self.MIN_NEGATIVES
        if self._uses_fallback:
            self._fallback.fit(index)
            return
        self._index = index
        self.w, self.boundary, self.width = self._train(self._positives(index), self._negatives(index))

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
