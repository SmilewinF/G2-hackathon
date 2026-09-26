"""Modifier signals: multiplicative vetoes and adjustments applied to novelty."""

from __future__ import annotations

from ..index import ARTICLE_ID, Analysis, ReferenceIndex
from ..models import Stance
from ..text import containment, distinct_share, words
from .base import Kind, Signal, SignalResult


class DuplicateCheck(Signal):
    """Near-copy of an existing submission *or of the fixed content itself* → novelty 0.

    Character-shingle containment is model-independent and catches light edits, padding, sentence
    shuffles and stance-only changes. Normalisation in ``Submission`` defeats invisible-character
    and look-alike-letter evasion before this runs.
    """

    name = "duplicate"
    kind = Kind.MODIFIER

    def __init__(self, threshold: float = 0.6) -> None:
        self.threshold = threshold

    def evaluate(self, a: Analysis, index: ReferenceIndex) -> SignalResult:
        best_id, best = None, 0.0
        for e in index.entries:
            c = containment(a.shingles, e.shingles)
            if c > best:
                best_id, best = e.id, c
        if best >= self.threshold:
            what = "the article itself" if best_id == ARTICLE_ID else best_id
            return SignalResult(0.0, [f"near-copy of {what} ({best:.0%} shingle overlap)"], {"of": best_id})
        return SignalResult(1.0, [], {"of": None})


class ContentQuality(Signal):
    """No reward for text that makes no statement: keyword lists, word repetition, emoji walls."""

    name = "quality"
    kind = Kind.MODIFIER

    def __init__(self, min_distinct_share: float = 0.35) -> None:
        self.min_distinct_share = min_distinct_share

    def evaluate(self, a: Analysis, index: ReferenceIndex) -> SignalResult:
        flags = []
        if not any(a.substantive):
            flags.append("no substantive content (no sentence-like clause in the body)")
        if len(words(a.submission.body)) >= 12 and distinct_share(a.submission.body) < self.min_distinct_share:
            flags.append("repetitive text")
        return SignalResult(0.0 if flags else 1.0, flags, {"flags": flags})


class StanceRarity(Signal):
    """Small bonus for a less common stance. Multiplicative, so it can never rescue a copy."""

    name = "stance"
    kind = Kind.MODIFIER

    def __init__(self, weight: float = 0.1) -> None:
        self.weight = weight

    def fit(self, index: ReferenceIndex) -> None:
        counts = {s: 0 for s in Stance}
        for sub in index.submissions:
            counts[sub.stance] += 1
        total = sum(counts.values()) + len(Stance)
        self.p = {s: (c + 1) / total for s, c in counts.items()}  # Laplace-smoothed

    def rarity(self, stance: Stance) -> float:
        return 1.0 - self.p[stance] / max(self.p.values())

    def evaluate(self, a: Analysis, index: ReferenceIndex) -> SignalResult:
        r = self.rarity(a.submission.stance)
        return SignalResult(1.0 - self.weight + self.weight * r, [], {"rarity": r})
