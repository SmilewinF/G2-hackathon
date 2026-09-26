"""Novelty reward pipeline.

    score = novelty × relevance_gate            (both in [0, 1])

Novelty is measured *relative to the existing corpus*, using hybrid (embedding + TF-IDF)
similarity to the nearest existing submissions; relevance *relative to the topic* (the
fixed content plus accepted on-topic responses). Both are self-calibrating: instead of hard-coded
cosine thresholds (which differ per embedding model), each raw signal is normalised against a
reference distribution computed from the data itself, so the pipeline behaves the same on the
local model or on Gemini.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .embeddings import Embedder
from .lexical import TfidfIndex, containment, shingles
from .models import FixedContent, Neighbor, ScoreBreakdown, Stance, Submission


@dataclass(frozen=True)
class ScorerConfig:
    k: int = 5  # neighbourhood size for the local-density term
    dense_weight: float = 0.6  # hybrid similarity = w * embedding cosine + (1 - w) * TF-IDF cosine
    nearest_weight: float = 0.5  # blend of nearest-neighbour distance vs. mean top-k distance
    stance_weight: float = 0.1  # max share of novelty that stance rarity can move
    relevance_floor: float = 0.1  # calibrated relevance at/below which reward is 0
    relevance_full: float = 0.5  # calibrated relevance at/above which the gate is fully open
    duplicate_containment: float = 0.6  # shingle containment at which text counts as a copy


def _normal_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _smoothstep(lo: float, hi: float, x: float) -> float:
    t = min(max((x - lo) / (hi - lo), 0.0), 1.0)
    return t * t * (3.0 - 2.0 * t)


def _unit(v: np.ndarray) -> np.ndarray:
    return v / max(float(np.linalg.norm(v)), 1e-12)


class NoveltyScorer:
    def __init__(
        self,
        fixed: FixedContent,
        corpus: Sequence[Submission],
        embedder: Embedder,
        off_topic_anchors: Sequence[str],
        config: ScorerConfig = ScorerConfig(),
    ) -> None:
        if len(corpus) <= config.k:
            raise ValueError(f"corpus needs more than k={config.k} submissions")
        self.fixed = fixed
        self.embedder = embedder
        self.config = config

        vecs = embedder.embed([fixed.embedding_text, *off_topic_anchors])
        self._fixed_vec = vecs[0]
        # Direction of "generic comment chatter" for this model. Subtracting similarity to it
        # cancels the genre/format similarity every comment shares with every other comment.
        self._generic_vec = _unit(vecs[1:].mean(axis=0))

        self._corpus: list[Submission] = []
        self._shingles: list[frozenset[str]] = []
        self._topic_member: list[bool] = []
        self._tfidf = TfidfIndex()
        self._matrix = np.empty((0, vecs.shape[1]), dtype=np.float32)
        self._add_many(list(corpus), on_topic=True)

    # ------------------------------------------------------------------ corpus management

    @property
    def corpus(self) -> list[Submission]:
        return list(self._corpus)

    def add(self, sub: Submission) -> None:
        """Add a trusted, on-topic submission to the reference corpus."""
        self._add_many([sub], on_topic=True)

    def submit(self, sub: Submission) -> ScoreBreakdown:
        """Score against everything seen so far, then add it to the corpus.

        Every submission counts for future novelty comparisons, but only relevant ones join the
        topic reference, so a flood of off-topic spam cannot drag the notion of "on topic".
        """
        result = self.score(sub)
        self._add_many([sub], on_topic=result.relevance_gate > 0.0)
        return result

    def _add_many(self, subs: list[Submission], on_topic: bool) -> None:
        vecs = self.embedder.embed([s.text for s in subs])
        self._corpus.extend(subs)
        self._shingles.extend(shingles(s.text) for s in subs)
        self._topic_member.extend([on_topic] * len(subs))
        self._tfidf.add([s.text for s in subs])
        self._matrix = np.vstack([self._matrix, vecs])
        self._calibrate()

    def _calibrate(self) -> None:
        """Recompute the reference distributions that make raw signals comparable."""
        # Novelty: leave-one-out raw novelty of every corpus item vs. the rest. Median/MAD are
        # robust to the corpus's own outliers and near-duplicates.
        sims = self._hybrid(self._matrix @ self._matrix.T, self._tfidf.pairwise())
        np.fill_diagonal(sims, -np.inf)
        loo = np.array([self._raw_novelty(row) for row in sims])
        self._loo_raw = loo
        self._nov_median = float(np.median(loo))
        mad = float(np.median(np.abs(loo - self._nov_median))) * 1.4826
        self._nov_scale = max(mad, 1e-4)

        # Relevance: topic = fixed content + on-topic submissions. The typical member's
        # leave-one-out margin defines "fully relevant"; a margin of 0 defines "irrelevant".
        members = self._matrix[np.array(self._topic_member)]
        self._topic_sum = members.sum(axis=0) + self._fixed_vec
        loo_margins = [self._margin(v, _unit(self._topic_sum - v)) for v in members]
        self._on_topic_margin = max(float(np.median(loo_margins)), 1e-6)

        counts = {s: 0 for s in Stance}
        for sub in self._corpus:
            counts[sub.stance] += 1
        total = len(self._corpus) + len(Stance)
        self._stance_p = {s: (c + 1) / total for s, c in counts.items()}  # Laplace-smoothed

    # ------------------------------------------------------------------ signals

    def _hybrid(self, dense: np.ndarray, lexical: np.ndarray) -> np.ndarray:
        w = self.config.dense_weight
        return w * dense + (1.0 - w) * lexical

    def _raw_novelty(self, sims: np.ndarray) -> float:
        top = np.sort(sims)[::-1][: self.config.k]
        w = self.config.nearest_weight
        return w * (1.0 - float(top[0])) + (1.0 - w) * (1.0 - float(np.mean(top)))

    def _stance_rarity(self, stance: Stance) -> float:
        p_max = max(self._stance_p.values())
        return 1.0 - self._stance_p[stance] / p_max

    def _margin(self, vec: np.ndarray, topic: np.ndarray) -> float:
        """How much closer the text is to the topic than to generic off-topic chatter."""
        return float(vec @ topic) - float(vec @ self._generic_vec)

    def _relevance(self, vec: np.ndarray) -> tuple[float, float]:
        margin = self._margin(vec, _unit(self._topic_sum))
        return min(max(margin / self._on_topic_margin, 0.0), 1.0), margin

    # ------------------------------------------------------------------ scoring

    def score(self, sub: Submission) -> ScoreBreakdown:
        cfg = self.config
        vec = self.embedder.embed([sub.text])[0]
        sims = self._hybrid(self._matrix @ vec, self._tfidf.similarities(sub.text))
        order = np.argsort(sims)[::-1][: cfg.k]
        nearest = [Neighbor(self._corpus[i].id, round(float(sims[i]), 4)) for i in order]
        reasons: list[str] = []

        raw = self._raw_novelty(sims)
        z = (raw - self._nov_median) / self._nov_scale
        semantic = _normal_cdf(z)

        sub_shingles = shingles(sub.text)
        overlaps = [containment(sub_shingles, s) for s in self._shingles]
        best = int(np.argmax(overlaps))
        duplicate_of = None
        if overlaps[best] >= cfg.duplicate_containment:
            duplicate_of = self._corpus[best].id
            semantic = 0.0
            reasons.append(f"near-copy of {duplicate_of} ({overlaps[best]:.0%} shingle overlap)")
        else:
            reasons.append(f"novelty z={z:+.2f} vs. corpus (nearest {nearest[0].id} @ {nearest[0].similarity:.2f})")

        rarity = self._stance_rarity(sub.stance)
        novelty = semantic * (1.0 - cfg.stance_weight + cfg.stance_weight * rarity)

        relevance, margin = self._relevance(vec)
        gate = _smoothstep(cfg.relevance_floor, cfg.relevance_full, relevance)
        score = novelty * gate

        if gate == 0.0:
            reasons.append(f"relevance {relevance:.2f} (margin {margin:+.3f}) is below floor: not rewarded")
        elif gate < 1.0:
            reasons.append(f"relevance {relevance:.2f} partially gates the reward ({gate:.2f})")

        return ScoreBreakdown(
            score=round(score, 4),
            novelty=round(novelty, 4),
            semantic_novelty=round(semantic, 4),
            raw_novelty=round(raw, 4),
            stance_rarity=round(rarity, 4),
            relevance=round(relevance, 4),
            relevance_margin=round(margin, 4),
            relevance_gate=round(gate, 4),
            near_duplicate_of=duplicate_of,
            nearest=nearest,
            reasons=reasons,
        )

    def corpus_novelty(self) -> dict[str, float]:
        """Leave-one-out semantic novelty of each corpus item (useful for inspection)."""
        return {
            s.id or str(i): round(_normal_cdf((r - self._nov_median) / self._nov_scale), 4)
            for i, (s, r) in enumerate(zip(self._corpus, self._loo_raw))
        }

    def corpus_relevance(self) -> dict[str, float]:
        """Leave-one-out calibrated relevance of each on-topic corpus item."""
        out = {}
        for i, (s, v) in enumerate(zip(self._corpus, self._matrix)):
            if self._topic_member[i]:
                m = self._margin(v, _unit(self._topic_sum - v))
                out[s.id or str(i)] = round(min(max(m / self._on_topic_margin, 0.0), 1.0), 4)
        return out
