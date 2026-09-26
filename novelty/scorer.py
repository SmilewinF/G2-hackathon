"""Novelty reward pipeline.

    score = novelty × relevance_gate            (both in [0, 1])

Novelty is measured *relative to the existing corpus* and relevance *relative to the fixed
content*. Both are self-calibrating: instead of hard-coded cosine thresholds (which differ per
embedding model), each raw signal is normalised against a reference distribution computed from
the data itself. That keeps the pipeline stable whether it runs on the local model or Gemini.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .embeddings import Embedder
from .lexical import containment, shingles
from .models import FixedContent, Neighbor, ScoreBreakdown, Stance, Submission


@dataclass(frozen=True)
class ScorerConfig:
    k: int = 5  # neighbourhood size for the local-density term
    nearest_weight: float = 0.5  # blend of nearest-neighbour distance vs. mean top-k distance
    stance_weight: float = 0.1  # max share of novelty that stance rarity can move
    relevance_floor: float = 0.3  # calibrated relevance at/below which reward is 0
    relevance_full: float = 0.6  # calibrated relevance at/above which the gate is fully open
    duplicate_containment: float = 0.6  # shingle containment at which text counts as a copy


def _normal_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _smoothstep(lo: float, hi: float, x: float) -> float:
    t = min(max((x - lo) / (hi - lo), 0.0), 1.0)
    return t * t * (3.0 - 2.0 * t)


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
        # Relevance floor reference: how similar clearly unrelated comments look to this model.
        self._off_topic_sim = float(np.mean(vecs[1:] @ self._fixed_vec))

        self._corpus: list[Submission] = []
        self._shingles: list[frozenset[str]] = []
        self._matrix = np.empty((0, vecs.shape[1]), dtype=np.float32)
        self._add_many(list(corpus))

    # ------------------------------------------------------------------ corpus management

    @property
    def corpus(self) -> list[Submission]:
        return list(self._corpus)

    def add(self, sub: Submission) -> None:
        self._add_many([sub])

    def submit(self, sub: Submission) -> ScoreBreakdown:
        """Score a submission against everything seen so far, then add it to the corpus."""
        result = self.score(sub)
        self.add(sub)
        return result

    def _add_many(self, subs: list[Submission]) -> None:
        vecs = self.embedder.embed([s.text for s in subs])
        self._corpus.extend(subs)
        self._shingles.extend(shingles(s.text) for s in subs)
        self._matrix = np.vstack([self._matrix, vecs])
        self._calibrate()

    def _calibrate(self) -> None:
        """Recompute the reference distributions that make raw signals comparable."""
        # Novelty: leave-one-out raw novelty of every corpus item vs. the rest. Median/MAD are
        # robust to the corpus's own outliers and near-duplicates.
        sims = self._matrix @ self._matrix.T
        np.fill_diagonal(sims, -np.inf)
        loo = np.array([self._raw_novelty(row) for row in sims])
        self._loo_raw = loo
        self._nov_median = float(np.median(loo))
        mad = float(np.median(np.abs(loo - self._nov_median))) * 1.4826
        self._nov_scale = max(mad, 1e-4)

        # Relevance: the typical on-topic submission defines "fully relevant".
        self._on_topic_sim = float(np.median(self._matrix @ self._fixed_vec))

        counts = {s: 0 for s in Stance}
        for sub in self._corpus:
            counts[sub.stance] += 1
        total = len(self._corpus) + len(Stance)
        self._stance_p = {s: (c + 1) / total for s, c in counts.items()}  # Laplace-smoothed

    # ------------------------------------------------------------------ scoring

    def _raw_novelty(self, sims: np.ndarray) -> float:
        k = self.config.k
        top = np.sort(sims)[::-1][:k]
        w = self.config.nearest_weight
        return w * (1.0 - float(top[0])) + (1.0 - w) * (1.0 - float(np.mean(top)))

    def _stance_rarity(self, stance: Stance) -> float:
        p_max = max(self._stance_p.values())
        return 1.0 - self._stance_p[stance] / p_max

    def _relevance(self, sim: float) -> float:
        span = max(self._on_topic_sim - self._off_topic_sim, 1e-6)
        return min(max((sim - self._off_topic_sim) / span, 0.0), 1.0)

    def score(self, sub: Submission) -> ScoreBreakdown:
        cfg = self.config
        vec = self.embedder.embed([sub.text])[0]
        sims = self._matrix @ vec
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

        rarity = self._stance_rarity(sub.stance)
        novelty = semantic * (1.0 - cfg.stance_weight + cfg.stance_weight * rarity)

        relevance = self._relevance(float(vec @ self._fixed_vec))
        gate = _smoothstep(cfg.relevance_floor, cfg.relevance_full, relevance)
        score = novelty * gate

        if duplicate_of is None:
            reasons.append(f"novelty z={z:+.2f} vs. corpus (nearest {nearest[0].id} @ {nearest[0].similarity:.2f})")
        if gate == 0.0:
            reasons.append(f"relevance {relevance:.2f} is below floor {cfg.relevance_floor}: not rewarded")
        elif gate < 1.0:
            reasons.append(f"relevance {relevance:.2f} partially gates the reward ({gate:.2f})")

        return ScoreBreakdown(
            score=round(score, 4),
            novelty=round(novelty, 4),
            semantic_novelty=round(semantic, 4),
            raw_novelty=round(raw, 4),
            stance_rarity=round(rarity, 4),
            relevance=round(relevance, 4),
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
