"""Reference index: everything a new submission is compared against, embedded once.

Holds the fixed content (as a reference-only entry, so restating the article is not novel) and
every admitted submission, with their dense vectors, TF-IDF rows, shingles and clauses. Signals
read from it; only ``NoveltyScorer`` writes to it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .embeddings import Embedder
from .lexical import TfidfIndex
from .models import FixedContent, Submission
from .preparation import EnglishPreparer, PreparedText, TextPreparer
from .text import shingles, tokens

ARTICLE_ID = "article"
FULL_LEXICAL_TOKENS = 6  # content tokens at which TF-IDF evidence gets its full hybrid weight (swept: 4-10)


def unit(v: np.ndarray) -> np.ndarray:
    return v / max(float(np.linalg.norm(v)), 1e-12)


@dataclass(frozen=True)
class Entry:
    id: str
    text: str  # prepared analysis text (normalised, spell-corrected, English only), not the display text
    submission: Submission | None  # None for the fixed content
    on_topic: bool  # contributes to the topic centroid
    shingles: frozenset[str]
    clauses: tuple[str, ...]
    substantive: tuple[bool, ...]

    @property
    def is_submission(self) -> bool:
        return self.submission is not None


@dataclass(frozen=True)
class Analysis:
    """Everything the signals need about one incoming submission, computed once."""

    submission: Submission
    prepared: PreparedText
    text: str  # prepared text
    vec: np.ndarray  # dense, whole text
    sims: np.ndarray  # hybrid similarity to every entry
    content_vec: np.ndarray | None  # dense, substantive body clauses only (None if there are none)
    clauses: tuple[str, ...]
    substantive: tuple[bool, ...]
    foreign: tuple[bool, ...]
    clause_sims: np.ndarray  # (n_clauses, n_entries) hybrid similarity
    clause_margins: np.ndarray  # topic margin per clause
    shingles: frozenset[str]


def _content_text(clause_list: Sequence[str], substantive: Sequence[bool]) -> str | None:
    kept = [c for c, ok in zip(clause_list, substantive) if ok]
    return ". ".join(kept) if kept else None


class ReferenceIndex:
    def __init__(
        self,
        fixed: FixedContent,
        embedder: Embedder,
        off_topic_anchors: Sequence[str],
        dense_weight: float,
        preparer: TextPreparer | None = None,
    ) -> None:
        if not off_topic_anchors:
            raise ValueError("at least one off-topic anchor is required to calibrate relevance")
        self.fixed = fixed
        self.embedder = embedder
        self.dense_weight = dense_weight
        self.preparer = preparer or EnglishPreparer([fixed.embedding_text])
        # Direction of "generic comment chatter" for this model. Subtracting similarity to it
        # cancels the genre/format similarity every comment shares with every other comment.
        self.generic_vec = unit(embedder.embed(list(off_topic_anchors)).mean(axis=0))

        self.entries: list[Entry] = []
        self._tfidf = TfidfIndex()
        self._dense = np.zeros((0, self.generic_vec.shape[0]), dtype=np.float32)
        self._content = np.zeros_like(self._dense)
        self._clause_vecs: list[np.ndarray] = []

        prep = self.preparer.prepare(fixed.title, fixed.text)
        self._append([Entry(ARTICLE_ID, prep.analysis_text, None, True, shingles(prep.text), prep.clauses, prep.substantive)])

    # ------------------------------------------------------------------ writing

    def add(self, subs: Sequence[Submission], on_topic: bool = True) -> None:
        entries = []
        for sub in subs:
            if sub.id is None:
                raise ValueError("submissions added to the index need an id")
            if any(e.id == sub.id for e in self.entries):
                raise ValueError(f"duplicate submission id {sub.id!r}")
            prep = self.preparer.prepare(sub.headline, sub.body)
            entries.append(Entry(sub.id, prep.analysis_text, sub, on_topic, shingles(prep.text), prep.clauses, prep.substantive))
        self._append(entries)

    def _append(self, entries: list[Entry]) -> None:
        batch, spans = [], []
        for e in entries:
            content = _content_text(e.clauses, e.substantive) or e.text
            start = len(batch)
            batch.extend([e.text, content, *e.clauses])
            spans.append(start)
        vecs = self.embedder.embed(batch) if batch else np.zeros((0, self._dense.shape[1]))
        for e, start in zip(entries, spans):
            self._clause_vecs.append(vecs[start + 2 : start + 2 + len(e.clauses)])
        self._dense = np.vstack([self._dense, vecs[[s for s in spans]]])
        self._content = np.vstack([self._content, vecs[[s + 1 for s in spans]]])
        self._tfidf.add([e.text for e in entries])
        self.entries.extend(entries)

    # ------------------------------------------------------------------ reading

    def __len__(self) -> int:
        return len(self.entries)

    @property
    def submission_indices(self) -> np.ndarray:
        return np.array([i for i, e in enumerate(self.entries) if e.is_submission], dtype=int)

    @property
    def submissions(self) -> list[Submission]:
        return [e.submission for e in self.entries if e.submission is not None]

    def hybrid(self, dense: np.ndarray, lexical: np.ndarray, lexical_confidence: float = 1.0) -> np.ndarray:
        """``lexical_confidence`` < 1 shifts weight to the dense term for very short queries,
        where a single rare word would otherwise dominate the TF-IDF cosine."""
        lw = (1.0 - self.dense_weight) * lexical_confidence
        return (1.0 - lw) * dense + lw * lexical

    def pairwise(self) -> np.ndarray:
        return self.hybrid(self._dense @ self._dense.T, self._tfidf.pairwise())

    def clause_sims_of(self, i: int) -> np.ndarray:
        """Hybrid similarity of entry ``i``'s clauses to every entry (its own column included)."""
        e = self.entries[i]
        return self.hybrid(self._clause_vecs[i] @ self._dense.T, self._tfidf.similarities_many(e.clauses))

    def content_vec(self, i: int) -> np.ndarray:
        return self._content[i]

    def topic_sum(self) -> np.ndarray:
        """Sum of the content vectors of on-topic entries (the fixed content is always one)."""
        mask = np.array([e.on_topic for e in self.entries])
        return self._content[mask].sum(axis=0)

    def margin(self, vecs: np.ndarray, topic: np.ndarray) -> np.ndarray:
        """How much closer each vector is to the topic than to generic off-topic chatter."""
        return vecs @ unit(topic) - vecs @ self.generic_vec

    def analyze(self, sub: Submission) -> Analysis:
        prep = self.preparer.prepare(sub.headline, sub.body)
        cl, subst, text = prep.clauses, prep.substantive, prep.analysis_text
        content = _content_text(cl, subst)
        batch = [text, *([content] if content else []), *cl]
        vecs = self.embedder.embed(batch)
        vec = vecs[0]
        content_vec = vecs[1] if content else None
        clause_vecs = vecs[2 if content else 1 :]
        clause_sims = self.hybrid(clause_vecs @ self._dense.T, self._tfidf.similarities_many(cl))
        return Analysis(
            submission=sub,
            prepared=prep,
            text=text,
            vec=vec,
            sims=self.hybrid(self._dense @ vec, self._tfidf.similarities(text),
                             min(1.0, len(tokens(text)) / FULL_LEXICAL_TOKENS)),
            content_vec=content_vec,
            clauses=cl,
            substantive=subst,
            foreign=prep.foreign,
            clause_sims=clause_sims,
            clause_margins=self.margin(clause_vecs, self.topic_sum()) if len(cl) else np.zeros(0),
            shingles=shingles(prep.text),
        )
