"""TF-IDF index: the lexical half of hybrid similarity.

Small sentence embedders rate a reworded version of an idea (~0.84 cosine) barely closer than a
genuinely new idea on the same topic (~0.75), but the two share rare terms ("stormwater",
"basements") that IDF weighting surfaces.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Sequence

import numpy as np

from .text import tokens


class TfidfIndex:
    """Sublinear-TF, smoothed-IDF cosine similarity over a growing document set."""

    def __init__(self) -> None:
        self._docs: list[Counter[str]] = []
        self._matrix = np.zeros((0, 0))

    def __len__(self) -> int:
        return len(self._docs)

    def add(self, texts: Sequence[str]) -> None:
        self._docs.extend(Counter(tokens(t)) for t in texts)
        df: Counter[str] = Counter(t for doc in self._docs for t in doc)
        n = len(self._docs)
        self._vocab = {t: i for i, t in enumerate(df)}
        self._idf = np.array([math.log((n + 1) / (df[t] + 1)) + 1.0 for t in self._vocab])
        self._unseen_idf = math.log(n + 1) + 1.0
        self._matrix = np.vstack([self._vector(doc) for doc in self._docs])

    def _vector(self, counts: Counter[str]) -> np.ndarray:
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
        return v / norm if norm else v

    def pairwise(self) -> np.ndarray:
        return self._matrix @ self._matrix.T

    def similarities(self, text: str) -> np.ndarray:
        return self._matrix @ self._vector(Counter(tokens(text)))

    def similarities_many(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, len(self._docs)))
        return np.vstack([self.similarities(t) for t in texts])
