"""Reference index: everything a new submission is compared against, embedded once.

Holds the fixed content (as a reference-only entry, so restating the article is not novel) and
every admitted submission, with their dense vectors, sparse TF-IDF vectors, shingles and clauses.
Signals read from it; only ``NoveltyScorer`` writes to it.

Everything is computed once per text at insertion: preparation, tokenisation, embedding (in one
batch per call), sparse vectors. Dense matrices grow with capacity doubling instead of being
re-stacked on every insert. ``version`` changes only when the TF-IDF model is refit, which is
the one event after which previously computed lexical similarities are stale.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .embeddings import Embedder
from .errors import CalibrationError, ValidationError
from .lexical import SparseVec, SparseVectors, TfidfModel, count
from .models import FixedContent, Submission
from .preparation import EnglishPreparer, PreparedText, TextPreparer
from .text import shingles

log = logging.getLogger(__name__)

ARTICLE_ID = "article"
FULL_LEXICAL_TOKENS = 6  # content tokens at which TF-IDF evidence gets its full hybrid weight (swept: 4-10)


def unit(v: np.ndarray) -> np.ndarray:
    return v / max(float(np.linalg.norm(v)), 1e-12)


@dataclass(frozen=True)
class Entry:
    id: str
    text: str  # PreparedText.analysis_text, not the display text
    submission: Submission | None  # None for the fixed content
    on_topic: bool  # contributes to the topic centroid
    shingles: frozenset[str]
    clauses: tuple[str, ...]
    substantive: tuple[bool, ...]
    clause_rows: tuple[int, ...]  # rows of this entry's clauses in the index's clause matrices

    @property
    def is_submission(self) -> bool:
        return self.submission is not None


@dataclass(frozen=True)
class Analysis:
    """Everything the signals need about one incoming submission, computed once."""

    submission: Submission
    prepared: PreparedText
    text: str  # PreparedText.analysis_text, as stored in Entry.text
    vec: np.ndarray  # dense, whole text
    sims: np.ndarray  # hybrid similarity to every entry
    content_vec: np.ndarray | None  # dense, substantive body clauses only (None if there are none)
    clauses: tuple[str, ...]
    substantive: tuple[bool, ...]
    foreign: tuple[bool, ...]
    clause_sims: np.ndarray  # (n_clauses, n_entries) hybrid similarity
    clause_margins: np.ndarray  # topic margin per clause
    shingles: frozenset[str]
    clause_vecs: np.ndarray  # (n_clauses, d) dense, one per clause
    clause_relevance: np.ndarray | None = None  # per clause, set by the scorer from its relevance signals


def _content_text(clause_list: Sequence[str], substantive: Sequence[bool]) -> str | None:
    kept = [c for c, ok in zip(clause_list, substantive) if ok]
    return ". ".join(kept) if kept else None


class _Rows:
    """Growable float32 matrix with amortised O(1) appends."""

    def __init__(self, dim: int) -> None:
        self._data = np.zeros((16, dim), dtype=np.float32)
        self.n = 0

    def extend(self, rows: np.ndarray) -> None:
        need = self.n + len(rows)
        if need > len(self._data):
            grown = np.zeros((max(need, 2 * len(self._data)), self._data.shape[1]), dtype=np.float32)
            grown[: self.n] = self._data[: self.n]
            self._data = grown
        self._data[self.n : need] = rows
        self.n = need

    @property
    def view(self) -> np.ndarray:
        return self._data[: self.n]


class ReferenceIndex:
    def __init__(
        self,
        fixed: FixedContent,
        embedder: Embedder,
        off_topic_anchors: Sequence[str],
        dense_weight: float,
        preparer: TextPreparer | None = None,
        refit_growth: float = 0.1,
        relevance_negatives: Sequence[Submission] = (),
    ) -> None:
        if not off_topic_anchors:
            raise CalibrationError("at least one off-topic anchor is required to calibrate relevance")
        self.fixed = fixed
        self.embedder = embedder
        self.dense_weight = dense_weight
        self.preparer = preparer or EnglishPreparer([fixed.embedding_text])
        # Direction of "generic comment chatter" for this model. Subtracting similarity to it
        # cancels the genre/format similarity every comment shares with every other comment.
        anchor_vecs = embedder.embed(list(off_topic_anchors))
        self.generic_vec = unit(anchor_vecs.mean(axis=0))
        dim = anchor_vecs.shape[1]
        # Content vectors (prepared and embedded exactly like a submission's substantive body) of
        # the generic anchors and of the article-specific hard negatives: same-town comments that
        # are NOT about this article. The learned relevance signal trains on them.
        self.anchor_content = self._content_vectors([(".", a) for a in off_topic_anchors], dim)
        self.negative_content = self._content_vectors([(n.headline, n.body) for n in relevance_negatives], dim)

        self.entries: list[Entry] = []
        self._by_id: dict[str, int] = {}
        self._dense, self._content, self._clause_dense = _Rows(dim), _Rows(dim), _Rows(dim)
        self._on_topic: list[bool] = []
        self._topic_sum = np.zeros(dim, dtype=np.float64)
        # lexical state: counts are kept so a refit never re-tokenises
        self.tfidf = TfidfModel(refit_growth)
        self._doc_counts: list[Counter[str]] = []
        self._clause_counts: list[Counter[str]] = []
        self.clause_owner: list[int] = []  # entry index of each clause row
        self.clause_substantive: list[bool] = []
        self._doc_sparse = SparseVectors()
        self._clause_sparse = SparseVectors()

        prep = self.preparer.prepare(fixed.title, fixed.text)
        self._append([(ARTICLE_ID, None, prep, True)])

    def _content_vectors(self, pairs: Sequence[tuple[str, str]], dim: int) -> np.ndarray:
        if not pairs:
            return np.zeros((0, dim), dtype=np.float32)
        texts = []
        for headline, body in pairs:
            prep = self.preparer.prepare(headline, body)
            texts.append(_content_text(prep.clauses, prep.substantive) or prep.analysis_text)
        return self.embedder.embed(texts)

    # ------------------------------------------------------------------ writing

    def add(self, subs: Sequence[Submission], on_topic: bool = True) -> range:
        """Insert submissions; returns the range of their entry indices."""
        batch_ids: set[str] = set()
        for sub in subs:
            if sub.id is None:
                raise ValidationError("submissions added to the index need an id")
            if sub.id in self._by_id or sub.id in batch_ids:
                raise ValidationError(f"duplicate submission id {sub.id!r}")
            batch_ids.add(sub.id)
        start = len(self.entries)
        self._append([(sub.id, sub, self.preparer.prepare(sub.headline, sub.body), on_topic) for sub in subs])
        return range(start, len(self.entries))

    def _append(self, items: list[tuple[str, Submission | None, PreparedText, bool]]) -> None:
        batch, spans = [], []
        for _, _, prep, _ in items:
            content = _content_text(prep.clauses, prep.substantive) or prep.analysis_text
            spans.append(len(batch))
            batch.extend([prep.analysis_text, content, *prep.clauses])
        vecs = self.embedder.embed(batch)

        for (eid, sub, prep, on_topic), start in zip(items, spans):
            first_clause = len(self.clause_owner)
            n_cl = len(prep.clauses)
            idx = len(self.entries)
            self.entries.append(Entry(eid, prep.analysis_text, sub, on_topic, shingles(prep.text), prep.clauses,
                                      prep.substantive, tuple(range(first_clause, first_clause + n_cl))))
            self._by_id[eid] = idx
            self._dense.extend(vecs[start : start + 1])
            self._content.extend(vecs[start + 1 : start + 2])
            self._clause_dense.extend(vecs[start + 2 : start + 2 + n_cl])
            self._on_topic.append(on_topic)
            if on_topic:
                self._topic_sum += vecs[start + 1]
            doc = count(prep.analysis_text)
            self._doc_counts.append(doc)
            self.tfidf.observe(doc)
            for c, ok in zip(prep.clauses, prep.substantive):
                self._clause_counts.append(count(c))
                self.clause_owner.append(idx)
                self.clause_substantive.append(ok)

        self._sync_sparse()

    def _sync_sparse(self) -> None:
        """Vectorise the counts that have no sparse vector yet; after a TF-IDF refit, all of them."""
        if self.tfidf.needs_refit:
            self.tfidf.refit()
            log.debug("TF-IDF refit at %d entries (version %d)", len(self.entries), self.tfidf.version)
            self._doc_sparse, self._clause_sparse = SparseVectors(), SparseVectors()
        self._doc_sparse.extend(self.tfidf.vector(d) for d in self._doc_counts[len(self._doc_sparse):])
        self._clause_sparse.extend(self.tfidf.vector(c) for c in self._clause_counts[len(self._clause_sparse):])

    # ------------------------------------------------------------------ reading

    def __len__(self) -> int:
        return len(self.entries)

    def get(self, entry_id: str) -> Entry | None:
        i = self._by_id.get(entry_id)
        return None if i is None else self.entries[i]

    @property
    def version(self) -> int:
        """Changes whenever stored lexical vectors were rebuilt (TF-IDF refit)."""
        return self.tfidf.version

    @property
    def submission_indices(self) -> np.ndarray:
        return np.array([i for i, e in enumerate(self.entries) if e.is_submission], dtype=int)

    @property
    def submissions(self) -> list[Submission]:
        return [e.submission for e in self.entries if e.submission is not None]

    @property
    def dense(self) -> np.ndarray:
        return self._dense.view

    @property
    def clause_dense(self) -> np.ndarray:
        return self._clause_dense.view

    @property
    def content(self) -> np.ndarray:
        return self._content.view

    @property
    def on_topic_mask(self) -> np.ndarray:
        return np.array(self._on_topic, dtype=bool)

    def hybrid(self, dense: np.ndarray, lexical: np.ndarray, lexical_confidence: float = 1.0) -> np.ndarray:
        """Blend of embedding and TF-IDF similarity: ``dense_weight`` x dense + the rest x lexical.

        ``lexical_confidence`` < 1 shifts weight to the dense term for very short queries,
        where a single rare word would otherwise dominate the TF-IDF cosine."""
        lw = (1.0 - self.dense_weight) * lexical_confidence
        return (1.0 - lw) * dense + lw * lexical

    def doc_vector(self, i: int) -> SparseVec:
        return self._doc_sparse.vectors[i]

    def entry_sims(self, i: int) -> np.ndarray:
        """Hybrid similarity of stored entry ``i`` to every entry (itself included)."""
        return self.hybrid(self.dense @ self.dense[i], self._doc_sparse.dot_all(self._doc_sparse.vectors[i]))

    def clause_sims_to_entry(self, j: int) -> np.ndarray:
        """Hybrid similarity of every stored clause to entry ``j`` (one column of the clause matrix)."""
        return self.hybrid(self.clause_dense @ self.dense[j], self._clause_sparse.dot_all(self._doc_sparse.vectors[j]))

    def clause_sims(self, c: int) -> np.ndarray:
        """Hybrid similarity of stored clause row ``c`` to every entry."""
        return self.hybrid(self.dense @ self.clause_dense[c], self._doc_sparse.dot_all(self._clause_sparse.vectors[c]))

    def content_vec(self, i: int) -> np.ndarray:
        return self.content[i]

    def topic_sum(self) -> np.ndarray:
        """Sum of the content vectors of on-topic entries (the fixed content is always one)."""
        return self._topic_sum.astype(np.float32)

    def margin(self, vecs: np.ndarray, topic: np.ndarray) -> np.ndarray:
        """How much closer each vector is to the topic than to generic off-topic chatter."""
        return vecs @ unit(topic) - vecs @ self.generic_vec

    def analyze(self, sub: Submission) -> Analysis:
        """Prepare, embed and compare one submission with every entry, without adding it to the index."""
        prep = self.preparer.prepare(sub.headline, sub.body)
        cl, subst, text = prep.clauses, prep.substantive, prep.analysis_text
        content = _content_text(cl, subst)
        vecs = self.embedder.embed([text, *([content] if content else []), *cl])
        vec = vecs[0]
        clause_vecs = vecs[2 if content else 1 :]
        doc = count(text)
        lexical = self._doc_sparse.dot_all(self.tfidf.vector(doc))
        lexical_confidence = min(1.0, sum(doc.values()) / FULL_LEXICAL_TOKENS)
        clause_lex = [self._doc_sparse.dot_all(self.tfidf.vector(count(c))) for c in cl]
        dense = self.dense
        return Analysis(
            submission=sub,
            prepared=prep,
            text=text,
            vec=vec,
            sims=self.hybrid(dense @ vec, lexical, lexical_confidence),
            content_vec=vecs[1] if content else None,
            clauses=cl,
            substantive=subst,
            foreign=prep.foreign,
            clause_sims=(self.hybrid(clause_vecs @ dense.T, np.vstack(clause_lex)) if cl
                         else np.zeros((0, len(self.entries)))),
            clause_margins=self.margin(clause_vecs, self._topic_sum) if cl else np.zeros(0),
            shingles=shingles(prep.text),
            clause_vecs=clause_vecs if cl else np.zeros((0, dense.shape[1]), dtype=np.float32),
        )
