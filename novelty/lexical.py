"""Model-independent lexical signals.

- Character-shingle containment catches "same text": copy-paste with light edits, or a copy padded
  with extra words, regardless of which embedding backend is in use.
- A TF-IDF index over the corpus catches "same idea, same distinctive vocabulary". Small sentence
  embedders rate a reworded version of an idea (~0.84 cosine) barely closer than a genuinely new
  idea on the same topic (~0.75), but the two share rare terms ("stormwater", "basements"), which
  IDF weighting surfaces. The scorer blends this with dense similarity (hybrid similarity).
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Sequence

import numpy as np

_NON_WORD = re.compile(r"[^a-z0-9]+")
_WORD = re.compile(r"[a-z]+")
_STOP = frozenset(
    """a an the and or but if of to in on at by for with from as is are was were be been being it
    its this that these those i we you they he she my our your their me us them not no so do does
    did will would can could should just than then there here what who whom which when where why
    how all any some more most other such only own same too very about into over after before up
    down out off also has have had his her him""".split()
)


def shingles(text: str, n: int = 5) -> frozenset[str]:
    norm = _NON_WORD.sub(" ", text.lower()).strip()
    if len(norm) <= n:
        return frozenset({norm})
    return frozenset(norm[i : i + n] for i in range(len(norm) - n + 1))


def containment(a: frozenset[str], b: frozenset[str]) -> float:
    """Share of the smaller shingle set found in the larger one (1.0 = one text contains the other)."""
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


def _stem(w: str) -> str:
    if len(w) > 5 and w.endswith("ing"):
        return w[:-3]
    if len(w) > 4 and w.endswith("ed"):
        return w[:-2]
    if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
        return w[:-1]
    return w


def tokens(text: str) -> list[str]:
    return [_stem(w) for w in _WORD.findall(text.lower()) if len(w) > 2 and w not in _STOP]


class TfidfIndex:
    """Sublinear-TF, smoothed-IDF cosine similarity over a growing document set."""

    def __init__(self) -> None:
        self._docs: list[Counter[str]] = []

    def add(self, texts: Sequence[str]) -> None:
        self._docs.extend(Counter(tokens(t)) for t in texts)
        df: Counter[str] = Counter(t for doc in self._docs for t in doc)
        n = len(self._docs)
        self._vocab = {t: i for i, t in enumerate(df)}
        self._idf = np.array([math.log((n + 1) / (df[t] + 1)) + 1.0 for t in self._vocab])
        self._unseen_idf = math.log(n + 1) + 1.0
        self._matrix = np.vstack([self._vector(doc)[0] for doc in self._docs])

    def _vector(self, counts: Counter[str]) -> tuple[np.ndarray, float]:
        """Unit vector over the known vocabulary. Unknown terms still count toward the norm, so a
        text full of words the corpus has never used is correctly far from everything."""
        v = np.zeros(len(self._vocab))
        unseen_sq = 0.0
        for t, c in counts.items():
            w = 1.0 + math.log(c)
            if t in self._vocab:
                v[self._vocab[t]] = w * self._idf[self._vocab[t]]
            else:
                unseen_sq += (w * self._unseen_idf) ** 2
        norm = math.sqrt(float(v @ v) + unseen_sq)
        return (v / norm if norm else v), norm

    def pairwise(self) -> np.ndarray:
        return self._matrix @ self._matrix.T

    def similarities(self, text: str) -> np.ndarray:
        return self._matrix @ self._vector(Counter(tokens(text)))[0]
