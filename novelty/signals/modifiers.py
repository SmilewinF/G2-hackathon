"""Modifier signals: multiplicative vetoes and adjustments applied to novelty."""

from __future__ import annotations

import numpy as np

from ..index import ARTICLE_ID, Analysis, ReferenceIndex
from ..models import Stance
from ..text import containment, distinct_share, specific_words, words
from .base import Kind, Signal, SignalResult


class DuplicateCheck(Signal):
    """Near-copy of an existing submission *or of the fixed content itself* → novelty 0.

    Character-shingle containment is model-independent and catches light edits, padding, sentence
    shuffles and stance-only changes. Normalisation in ``Submission`` defeats invisible-character
    and look-alike-letter evasion before this runs.

    An entry counts as contained when ``threshold`` of the smaller shingle set is in the larger
    one. The submission is a copy when, in addition, the entries it contains make up at least
    ``min_share`` of it: a copy with filler or several pasted comments is, a new argument that
    quotes a short comment is not (it used to score 0 as a "copy" of the quote).
    """

    name = "duplicate"
    kind = Kind.MODIFIER

    def __init__(self, threshold: float = 0.6, candidates: int = 25, min_share: float = 0.5) -> None:
        self.threshold = threshold
        self.candidates = candidates
        self.min_share = min_share

    def evaluate(self, a: Analysis, index: ReferenceIndex) -> SignalResult:
        # A copy is necessarily among the most similar entries, so only those (and the article)
        # need the exact shingle check: O(candidates) instead of O(corpus) set intersections.
        n = len(index.entries)
        if n > self.candidates:
            pool = set(np.argpartition(a.sims, -self.candidates)[-self.candidates:].tolist()) | {0}
        else:
            pool = range(n)
        best_id, best, copied = None, 0.0, set()
        for i in pool:
            e = index.entries[i]
            c = containment(a.shingles, e.shingles)
            if c > best:
                best_id, best = e.id, c
            if c >= self.threshold:
                copied |= a.shingles & e.shingles
        share = len(copied) / len(a.shingles) if a.shingles else 0.0
        if best >= self.threshold and share >= self.min_share:
            what = "the article itself" if best_id == ARTICLE_ID else best_id
            return SignalResult(0.0, [f"near-copy of {what} ({best:.0%} shingle overlap)"], {"of": best_id, "share": share})
        return SignalResult(1.0, [], {"of": None, "share": share})


class ContentQuality(Signal):
    """No reward for text that makes no statement or cannot be assessed: keyword lists, word
    repetition, emoji walls, text with no substantive English clause. Foreign clauses next to
    English content are only noted in the reasons, not penalised."""

    name = "quality"
    kind = Kind.MODIFIER

    def __init__(self, min_distinct_share: float = 0.35, min_repetition_words: int = 12) -> None:
        self.min_distinct_share = min_distinct_share
        self.min_repetition_words = min_repetition_words  # shorter bodies are too short to call repetitive

    def evaluate(self, a: Analysis, index: ReferenceIndex) -> SignalResult:
        flags, notes = [], []
        if a.foreign and all(a.foreign):
            flags.append("text is not in a supported language (English), so it cannot be assessed")
        elif not any(a.substantive) and any(a.foreign):
            flags.append(f"no assessable English content ({sum(a.foreign)} of {len(a.foreign)} clauses are not in English)")
        elif not any(a.substantive):
            flags.append("no substantive content (no sentence-like clause in the body)")
        elif any(a.foreign):
            notes.append(f"{sum(a.foreign)} of {len(a.foreign)} clauses are not in English and were not scored")
        if len(words(a.submission.body)) >= self.min_repetition_words and distinct_share(a.submission.body) < self.min_distinct_share:
            flags.append("repetitive text")
        return SignalResult(0.0 if flags else 1.0, flags + notes, {"flags": flags})


class Specificity(Signal):
    """Discount text that makes no concrete point ("I love this park idea!").

    Very short texts sit far from everything in embedding space, so generic praise looks
    "novel". Counting specific content words (not function words, not evaluative words) gives a
    length-independent measure of whether there is a point to be novel about. Below
    ``min_specific`` distinct specific words the reward scales down quadratically; a single
    concrete idea ("Put EV chargers at the outer shuttle lot") easily clears it.
    """

    name = "specificity"
    kind = Kind.MODIFIER

    def __init__(self, min_specific: int = 4) -> None:
        self.min_specific = min_specific

    def evaluate(self, a: Analysis, index: ReferenceIndex) -> SignalResult:
        english = [c for c, f in zip(a.clauses, a.foreign) if not f]
        specific = specific_words(" ".join([a.prepared.english_headline, *english]))
        value = min(1.0, len(specific) / self.min_specific) ** 2
        reasons = [] if value == 1.0 else [f"only {len(specific)} specific content word(s): generic comment discounted"]
        return SignalResult(value, reasons, {"specific_words": sorted(specific)})


class StanceRarity(Signal):
    """Small relative bonus for a less common stance: value = 1 − weight × (1 − rarity), so the
    most common stance is discounted by ``weight`` (10 % by default) and rarer ones by less.
    Multiplicative and never above 1, so it can never rescue a copy."""

    name = "stance"
    kind = Kind.MODIFIER
    p: dict[Stance, float]  # Laplace-smoothed stance frequencies, set by fit()

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
