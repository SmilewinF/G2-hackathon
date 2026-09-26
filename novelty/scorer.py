"""Novelty reward pipeline: orchestrates the reference index and the signals.

    novelty = min(novelty signals) × Π(modifier signals)
    score   = novelty × Π(relevance signals)                  (all in [0, 1])

Default signals (see ``novelty/signals``):
    novelty    whole_text       hybrid-similarity distance to nearest neighbours
               clause_coverage  most novel relevant, substantive clause (kitchen-sink / padding defence)
    modifier   duplicate        near-copy of a submission or of the article → 0
               quality          keyword lists, repetition, unsupported language → 0
               specificity      generic comments with no concrete point → scaled down
               stance           rarer stance, up to +10 %
    relevance  relevance        contrastive topic margin of the substantive body → smooth gate

Every signal self-calibrates against the corpus (no hard-coded cosine thresholds), so behaviour
carries across embedding backends. ``submit`` only admits relevant, non-duplicate, substantive
submissions into the reference corpus, so spam and copy floods cannot shift the calibration.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .embeddings import Embedder
from .index import ReferenceIndex
from .preparation import EnglishPreparer, TextPreparer
from .models import FixedContent, Neighbor, ScoreBreakdown, Submission
from .signals import (
    ClauseCoverage,
    ContentQuality,
    DuplicateCheck,
    Kind,
    Signal,
    SignalResult,
    Specificity,
    StanceRarity,
    TopicMargin,
    WholeTextNovelty,
)


@dataclass(frozen=True)
class ScorerConfig:
    k: int = 5  # neighbourhood size for the local-density term
    dense_weight: float = 0.6  # hybrid similarity = w * embedding cosine + (1 - w) * TF-IDF cosine
    nearest_weight: float = 0.5  # blend of nearest-neighbour distance vs. mean top-k distance
    stance_weight: float = 0.1  # max share of novelty that stance rarity can move
    relevance_floor: float = 0.1  # calibrated relevance at/below which reward is 0
    relevance_full: float = 0.5  # calibrated relevance at/above which the gate is fully open
    duplicate_containment: float = 0.6  # shingle containment at which text counts as a copy
    min_specific_words: int = 4  # distinct specific content words needed for full reward


def default_signals(config: ScorerConfig) -> list[Signal]:
    return [
        WholeTextNovelty(k=config.k, nearest_weight=config.nearest_weight),
        ClauseCoverage(),
        DuplicateCheck(threshold=config.duplicate_containment),
        ContentQuality(),
        Specificity(min_specific=config.min_specific_words),
        StanceRarity(weight=config.stance_weight),
        TopicMargin(floor=config.relevance_floor, full=config.relevance_full),
    ]


class NoveltyScorer:
    def __init__(
        self,
        fixed: FixedContent,
        corpus: Sequence[Submission],
        embedder: Embedder,
        off_topic_anchors: Sequence[str],
        config: ScorerConfig = ScorerConfig(),
        signals: Sequence[Signal] | None = None,
        preparer: TextPreparer | None = None,
    ) -> None:
        self.fixed = fixed
        self.embedder = embedder
        self.config = config
        self.signals = list(signals) if signals is not None else default_signals(config)
        names = [s.name for s in self.signals]
        if len(set(names)) != len(names):
            raise ValueError(f"signal names must be unique: {names}")
        if not any(s.kind is Kind.NOVELTY for s in self.signals):
            raise ValueError("at least one NOVELTY signal is required")
        # Article + seed corpus vocabulary: spelling correction prefers these words and never
        # "fixes" them (proper nouns, domain terms).
        preparer = preparer or EnglishPreparer([fixed.embedding_text, *(s.text for s in corpus)])
        self.index = ReferenceIndex(fixed, embedder, off_topic_anchors, config.dense_weight, preparer)
        seeded = [sub if sub.id else dataclasses.replace(sub, id=f"c{i + 1:02d}") for i, sub in enumerate(corpus)]
        self._add(seeded, on_topic=True)

    # ------------------------------------------------------------------ corpus management

    @property
    def corpus(self) -> list[Submission]:
        return self.index.submissions

    def signal(self, name: str) -> Signal:
        return next(s for s in self.signals if s.name == name)

    def add(self, sub: Submission, on_topic: bool = True) -> None:
        """Add a submission to the reference corpus without scoring it (e.g. replaying saved ones)."""
        self._add([sub], on_topic)

    def _add(self, subs: list[Submission], on_topic: bool) -> None:
        self.index.add(subs, on_topic=on_topic)
        for s in self.signals:
            s.fit(self.index)

    def admission(self, result: ScoreBreakdown) -> tuple[bool, str]:
        """Only content that could ever earn a reward becomes reference data."""
        if result.near_duplicate_of is not None:
            return False, f"not added: duplicate of {result.near_duplicate_of}"
        if result.signals.get("quality", 1.0) == 0.0:
            return False, "not added: no substantive content"
        if result.relevance_gate == 0.0:
            return False, "not added: off-topic"
        return True, "added to corpus"

    def submit(self, sub: Submission) -> ScoreBreakdown:
        """Score against everything seen so far; add it to the corpus if the admission policy allows."""
        result = self.score(sub)
        admitted, reason = self.admission(result)
        if admitted:
            if sub.id is None:
                sub = dataclasses.replace(sub, id=f"s{len(self.index) :03d}")
            self.add(sub)
        return dataclasses.replace(result, admitted=admitted, reasons=[*result.reasons, reason])

    # ------------------------------------------------------------------ scoring

    def score(self, sub: Submission) -> ScoreBreakdown:
        analysis = self.index.analyze(sub)
        results = {s.name: s.evaluate(analysis, self.index) for s in self.signals}
        return self._combine(results, analysis.sims)

    def _combine(self, results: dict[str, SignalResult], sims: np.ndarray) -> ScoreBreakdown:
        by_kind = {k: [(s.name, results[s.name]) for s in self.signals if s.kind is k] for k in Kind}
        novelty = min(r.value for _, r in by_kind[Kind.NOVELTY])
        for _, r in by_kind[Kind.MODIFIER]:
            novelty *= r.value
        gate = 1.0
        for _, r in by_kind[Kind.RELEVANCE]:
            gate *= r.value

        order = np.argsort(sims)[::-1][: self.config.k]
        nearest = [Neighbor(self.index.entries[i].id, round(float(sims[i]), 4)) for i in order]

        def get(name: str, key: str | None = None, default=0.0):
            r = results.get(name)
            if r is None:
                return default
            return r.value if key is None else r.detail.get(key, default)

        return ScoreBreakdown(
            score=round(novelty * gate, 4),
            novelty=round(novelty, 4),
            semantic_novelty=round(get("whole_text"), 4),
            clause_novelty=(round(get("clause_coverage"), 4)
                            if get("clause_coverage", "applied", False) else None),
            raw_novelty=round(get("whole_text", "raw"), 4),
            stance_rarity=round(get("stance", "rarity"), 4),
            relevance=round(get("relevance", "relevance", 1.0), 4),
            relevance_margin=None if get("relevance", "margin", None) is None else round(get("relevance", "margin"), 4),
            relevance_gate=round(gate, 4),
            near_duplicate_of=get("duplicate", "of", None),
            nearest=nearest,
            reasons=[reason for r in results.values() for reason in r.reasons],
            signals={name: round(r.value, 4) for name, r in results.items()},
            detail={name: r.detail for name, r in results.items()},
        )

    # ------------------------------------------------------------------ inspection

    def corpus_novelty(self) -> dict[str, float]:
        """Leave-one-out whole-text novelty of each corpus item."""
        sig = self.signal("whole_text")
        return {k: round(sig.scale(v), 4) for k, v in sig.loo.items()}

    def corpus_relevance(self) -> dict[str, float]:
        """Leave-one-out calibrated relevance of each on-topic corpus item."""
        return self.signal("relevance").corpus_relevance()
