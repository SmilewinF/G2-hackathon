"""Sparse TF-IDF: the lexical half of hybrid similarity.

Small sentence embedders rate a reworded version of an idea (~0.84 cosine) barely closer than a
genuinely new idea on the same topic (~0.75), but the two share rare terms ("stormwater",
"basements") that IDF weighting surfaces.

Performance design:
- Every text is tokenised once (callers keep the ``Counter``); vectors are sparse dicts.
- ``SparseVectors`` is an inverted index, so "similarity of a query to every stored vector"
  touches only the postings of the query's terms, not every document.
- ``TfidfModel`` freezes IDF between refits. New documents are vectorised with the current IDF;
  a full refit (``version`` bump) happens only once the corpus has grown by ``refit_growth``, so
  its O(n) rebuild is amortised to O(1) per insert.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Iterable

import numpy as np

from .text import tokens

SparseVec = dict[str, float]


def count(text: str) -> Counter[str]:
    return Counter(tokens(text))


class TfidfModel:
    """Document frequencies plus an IDF snapshot. Sublinear TF, smoothed IDF, unit-norm vectors."""

    def __init__(self, refit_growth: float = 0.1) -> None:
        self.refit_growth = refit_growth
        self.df: Counter[str] = Counter()
        self.n = 0
        self.version = 0
        self._idf: dict[str, float] = {}
        self._refit_at = 0

    def observe(self, doc: Counter[str]) -> None:
        self.df.update(doc.keys())
        self.n += 1

    @property
    def needs_refit(self) -> bool:
        return self.n >= self._refit_at

    def refit(self) -> None:
        n = self.n
        self._idf = {t: math.log((n + 1) / (c + 1)) + 1.0 for t, c in self.df.items()}
        self._refit_at = max(self.n + 1, math.ceil(self.n * (1.0 + self.refit_growth)))
        self.version += 1

    def idf(self, term: str) -> float:
        cached = self._idf.get(term)
        if cached is not None:
            return cached
        # Term first seen after the last refit (or never): compute from the live counts.
        return math.log((self.n + 1) / (self.df.get(term, 0) + 1)) + 1.0

    def vector(self, doc: Counter[str]) -> SparseVec:
        """Unit vector. Terms the corpus has never used still count toward the norm, so a text
        full of unseen words is correctly far from everything."""
        weights = {t: (1.0 + math.log(c)) * self.idf(t) for t, c in doc.items()}
        norm = math.sqrt(sum(w * w for w in weights.values()))
        if not norm:
            return {}
        return {t: w / norm for t, w in weights.items() if self.df.get(t, 0)}


class SparseVectors:
    """Append-only inverted index over sparse unit vectors."""

    def __init__(self) -> None:
        self._postings: dict[str, tuple[list[int], list[float]]] = {}
        self._arrays: dict[str, tuple[np.ndarray, np.ndarray]] = {}  # lazily frozen postings
        self.vectors: list[SparseVec] = []

    def __len__(self) -> int:
        return len(self.vectors)

    def add(self, vec: SparseVec) -> int:
        i = len(self.vectors)
        self.vectors.append(vec)
        for t, w in vec.items():
            ids, ws = self._postings.setdefault(t, ([], []))
            ids.append(i)
            ws.append(w)
            self._arrays.pop(t, None)
        return i

    def extend(self, vecs: Iterable[SparseVec]) -> None:
        for v in vecs:
            self.add(v)

    def _posting(self, term: str) -> tuple[np.ndarray, np.ndarray] | None:
        arr = self._arrays.get(term)
        if arr is None:
            p = self._postings.get(term)
            if p is None:
                return None
            arr = (np.fromiter(p[0], dtype=np.int64, count=len(p[0])), np.fromiter(p[1], dtype=np.float64, count=len(p[1])))
            self._arrays[term] = arr
        return arr

    def dot_all(self, query: SparseVec) -> np.ndarray:
        """Cosine of ``query`` with every stored vector (both unit-norm)."""
        out = np.zeros(len(self.vectors))
        for t, qw in query.items():
            p = self._posting(t)
            if p is not None:
                out[p[0]] += qw * p[1]
        return out


def dot(a: SparseVec, b: SparseVec) -> float:
    if len(a) > len(b):
        a, b = b, a
    return sum(w * b.get(t, 0.0) for t, w in a.items())
