"""Model-independent near-duplicate detection via character shingles.

Semantic embeddings judge "same idea"; this catches "same text" (copy-paste with light edits,
or a copy padded with extra words) regardless of which embedding backend is in use.
"""

from __future__ import annotations

import re

_NON_WORD = re.compile(r"[^a-z0-9]+")


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
