"""Novelty reward pipeline: orchestrates the reference index and the signals.

    novelty = min(novelty signals) × Π(modifier signals)
    score   = novelty × Π(relevance signals)                  (all in [0, 1])

Default signals (see ``novelty/signals``):
    novelty    whole_text       hybrid-similarity distance to nearest neighbours
               clause_coverage  most novel on-topic, substantive clause (kitchen-sink / padding defence)
    modifier   duplicate        near-copy of a submission or of the article → 0
               quality          keyword lists, repetition, unsupported language → 0
               specificity      generic comments with no concrete point → scaled down
               stance           discount by stance frequency, the most common −10 %
    relevance  relevance        learned topic discriminant on the substantive body → smooth gate
                                (the contrastive topic margin without article-specific negatives)

The similarity-based signals (whole_text, clause_coverage, relevance) self-calibrate against the
corpus (no hard-coded cosine thresholds), so behaviour carries across embedding backends; stance
uses the corpus stance frequencies, and the text checks (duplicate, quality, specificity) use
fixed, model-independent limits. ``submit`` only admits clearly on-topic, non-duplicate,
substantive submissions into the reference corpus (see ``admission``), so spam, copy floods and
off-topic text with one on-topic line cannot shift the calibration or the learned gate.

Logging (logger ``novelty.scorer``): construction writes one INFO "scorer ready" line; every
``score``/``submit`` writes two INFO lines, the input and the result with its calculation and
timing; DEBUG adds the full body, every signal, the nearest neighbours, clause counts and the
text-preparation corrections.
"""

from __future__ import annotations

import copy
import dataclasses
import logging
import time
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from .embeddings import Embedder
from .errors import NoveltyError, ScoringError, ValidationError
from .index import Analysis, ReferenceIndex
from .logging_setup import preview, printable
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
    TopicDiscriminant,
    WholeTextNovelty,
)

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ScorerConfig:
    k: int = 5  # neighbourhood size for the local-density term
    dense_weight: float = 0.6  # hybrid similarity = w * embedding cosine + (1 - w) * TF-IDF cosine
    nearest_weight: float = 0.5  # blend of nearest-neighbour distance vs. mean top-k distance
    stance_weight: float = 0.1  # max share of novelty that stance rarity can move
    relevance_floor: float = 0.1  # TopicMargin: calibrated relevance at/below which reward is 0
    relevance_full: float = 0.5  # TopicMargin: calibrated relevance at/above which the gate is fully open
    relevance_shrinkage: float = 0.9  # TopicDiscriminant: covariance shrinkage toward the identity
    relevance_band: float = 0.2  # TopicDiscriminant: gate half-width, as a share of the on-topic margin (dev split)
    clause_min_relevance: float = 0.5  # ClauseCoverage: a clause is on-topic above this relevance of its own (dev split)
    clause_min_specific: int = 2  # ClauseCoverage: specific content words a clause needs to supply the novelty
    duplicate_containment: float = 0.6  # shingle containment at which an entry counts as contained in the text
    duplicate_min_share: float = 0.5  # ...and the share of the text such entries must make up for it to be a copy
    min_specific_words: int = 4  # distinct specific content words needed for full reward
    admission_min_gate: float = 0.5  # submit(): relevance gate needed to join the corpus (0.5 = at the boundary)
    admission_min_on_topic_share: float = 0.2  # submit(): share of substantive words in on-topic clauses (dev split)


def default_signals(config: ScorerConfig) -> list[Signal]:
    """The standard signal set, parameterised by ``config``; extend the list to plug in more."""
    return [
        WholeTextNovelty(k=config.k, nearest_weight=config.nearest_weight),
        ClauseCoverage(min_relevance=config.clause_min_relevance, min_specific=config.clause_min_specific),
        DuplicateCheck(threshold=config.duplicate_containment, min_share=config.duplicate_min_share),
        ContentQuality(),
        Specificity(min_specific=config.min_specific_words),
        StanceRarity(weight=config.stance_weight),
        # Learned relevance when the scorer has article-specific hard negatives; TopicMargin otherwise.
        TopicDiscriminant(shrinkage=config.relevance_shrinkage, band=config.relevance_band,
                          floor=config.relevance_floor, full=config.relevance_full),
    ]


class NoveltyScorer:
    """Scores submissions against the fixed content and a growing reference corpus (see module docstring)."""

    def __init__(
        self,
        fixed: FixedContent,
        corpus: Sequence[Submission],
        embedder: Embedder,
        off_topic_anchors: Sequence[str],
        config: ScorerConfig = ScorerConfig(),
        signals: Sequence[Signal] | None = None,
        preparer: TextPreparer | None = None,
        relevance_negatives: Sequence[Submission] = (),
    ) -> None:
        start = time.perf_counter()
        self.fixed = fixed
        self.embedder = embedder
        self.config = config
        self.signals = list(signals) if signals is not None else default_signals(config)
        names = [s.name for s in self.signals]
        if len(set(names)) != len(names):
            raise ValidationError(f"signal names must be unique: {names}")
        if not any(s.kind is Kind.NOVELTY for s in self.signals):
            raise ValidationError("at least one NOVELTY signal is required")
        # Article + seed corpus vocabulary: spelling correction prefers these words and never
        # "fixes" them (proper nouns, domain terms).
        preparer = preparer or EnglishPreparer([fixed.embedding_text, *(s.text for s in corpus)])
        self.index = ReferenceIndex(fixed, embedder, off_topic_anchors, config.dense_weight, preparer,
                                    relevance_negatives=relevance_negatives)
        seeded = [sub if sub.id else dataclasses.replace(sub, id=f"c{i + 1:02d}") for i, sub in enumerate(corpus)]
        self._add(seeded, on_topic=True)
        self._log_ready(time.perf_counter() - start)

    # ------------------------------------------------------------------ corpus management

    @property
    def corpus(self) -> list[Submission]:
        return self.index.submissions

    def fork(self) -> NoveltyScorer:
        """Independent copy of the corpus and calibration state (~3x cheaper than rebuilding).

        The embedder (model session, cache connection) and the text preparer (spellchecker) are
        stateless from the scorer's point of view and are shared, not copied.
        """
        shared = {id(self.embedder): self.embedder, id(self.index.preparer): self.index.preparer}
        return copy.deepcopy(self, shared)

    def signal(self, name: str) -> Signal:
        """The configured signal called ``name``; raises StopIteration if there is none."""
        return next(s for s in self.signals if s.name == name)

    def add(self, sub: Submission, on_topic: bool = True) -> None:
        """Add a submission to the reference corpus without scoring it."""
        self._add([sub], on_topic)

    def add_many(self, subs: Sequence[Submission], on_topic: bool = True) -> None:
        """Add several submissions with a single recalibration (e.g. replaying saved ones)."""
        if subs:
            self._add(list(subs), on_topic)

    def _add(self, subs: list[Submission], on_topic: bool) -> None:
        added = self.index.add(subs, on_topic=on_topic)
        for s in self.signals:
            try:
                s.update(self.index, added)
            except NoveltyError:
                raise
            except Exception:
                # An incremental update is an optimisation; a full fit gives the same state.
                log.exception("signal %r failed to update incrementally; recalibrating from scratch", s.name)
                try:
                    s.fit(self.index)
                except Exception as e:
                    raise ScoringError(f"signal {s.name!r} could not recalibrate after adding "
                                       f"{len(subs)} submission(s): {e}") from e
        log.debug("added %d submission(s); corpus now %d", len(added), len(self.corpus))

    def admission(self, result: ScoreBreakdown) -> tuple[bool, str]:
        """Only clearly on-topic content that could earn a reward becomes reference data. Admitted
        comments are positives for the learned relevance gate, so borderline ones are kept out:
        admitting everything with a gate above 0 let off-topic text with one appended garage line
        in: on dev, the first four of those moved the gate's boundary from 19.3 to 30.6."""
        if result.near_duplicate_of is not None:
            return False, f"not added: duplicate of {result.near_duplicate_of}"
        if result.signals.get("quality", 1.0) == 0.0:
            return False, "not added: no substantive content"
        if result.relevance_gate < self.config.admission_min_gate:
            return False, "not added: off-topic"
        share = result.detail.get("clause_coverage", {}).get("on_topic_share")
        if share is not None and share < self.config.admission_min_on_topic_share:
            return False, f"not added: mostly off-topic ({share:.0%} of the content is on-topic)"
        return True, "added to corpus"

    def submit(self, sub: Submission) -> ScoreBreakdown:
        """Score against everything seen so far; add it to the corpus if the admission policy allows."""
        start, stats = time.perf_counter(), self._embed_stats()
        self._log_input("submit", sub)
        result = self._score(sub)
        admitted, reason = self.admission(result)
        if admitted:
            if sub.id is None:
                sub = dataclasses.replace(sub, id=self._free_id("s"))
            self.add(sub)
        result = dataclasses.replace(result, admitted=admitted, reasons=[*result.reasons, reason])
        self._log_result("submit", sub, result, start, stats, reason)
        return result

    def _free_id(self, prefix: str) -> str:
        n = len(self.index)
        while self.index.get(f"{prefix}{n:03d}") is not None:
            n += 1
        return f"{prefix}{n:03d}"

    # ------------------------------------------------------------------ scoring

    def score(self, sub: Submission) -> ScoreBreakdown:
        """Score against the current corpus without changing it (``submit`` also applies admission)."""
        start, stats = time.perf_counter(), self._embed_stats()
        self._log_input("score", sub)
        result = self._score(sub)
        self._log_result("score", sub, result, start, stats)
        return result

    def _score(self, sub: Submission) -> ScoreBreakdown:
        analysis = self._with_clause_relevance(self.index.analyze(sub))
        results = {}
        for s in self.signals:
            try:
                results[s.name] = s.evaluate(analysis, self.index)
            except NoveltyError:
                raise
            except Exception as e:
                raise ScoringError(f"signal {s.name!r} failed: {e}") from e
        result = self._combine(results, analysis.sims)
        if log.isEnabledFor(logging.DEBUG):
            self._log_calculation(result, analysis)
        return result

    def _with_clause_relevance(self, analysis: Analysis) -> Analysis:
        """Attach each clause's own relevance (lowest across the relevance signals that judge
        clauses), so novelty signals can tell on-topic clauses from off-topic ones."""
        views = []
        for s in self.signals:
            if s.kind is not Kind.RELEVANCE:
                continue
            try:
                view = s.clause_relevance(analysis, self.index)
            except NoveltyError:
                raise
            except Exception as e:
                raise ScoringError(f"signal {s.name!r} failed to judge clauses: {e}") from e
            if view is not None:
                views.append(np.asarray(view, dtype=float))
        if not views:
            return analysis
        return dataclasses.replace(analysis, clause_relevance=np.minimum.reduce(views))

    def _combine(self, results: dict[str, SignalResult], sims: np.ndarray) -> ScoreBreakdown:
        values = {k: [results[s.name].value for s in self.signals if s.kind is k] for k in Kind}
        novelty = min(values[Kind.NOVELTY])
        for v in values[Kind.MODIFIER]:
            novelty *= v
        gate = 1.0
        for v in values[Kind.RELEVANCE]:
            gate *= v

        order = np.argsort(sims)[::-1][: self.config.k]
        nearest = [Neighbor(self.index.entries[i].id, round(float(sims[i]), 4)) for i in order]

        def get(name: str, key: str | None = None, default: Any = 0.0) -> Any:
            r = results.get(name)
            if r is None:
                return default
            return r.value if key is None else r.detail.get(key, default)

        margin = get("relevance", "margin", None)
        return ScoreBreakdown(
            score=round(novelty * gate, 4),
            novelty=round(novelty, 4),
            semantic_novelty=round(get("whole_text"), 4),
            clause_novelty=(round(get("clause_coverage"), 4)
                            if get("clause_coverage", "applied", False) else None),
            raw_novelty=round(get("whole_text", "raw"), 4),
            stance_rarity=round(get("stance", "rarity"), 4),
            relevance=round(get("relevance", "relevance", 1.0), 4),
            relevance_margin=None if margin is None else round(margin, 4),
            relevance_gate=round(gate, 4),
            near_duplicate_of=get("duplicate", "of", None),
            nearest=nearest,
            reasons=[reason for r in results.values() for reason in r.reasons],
            signals={name: round(r.value, 4) for name, r in results.items()},
            detail={name: r.detail for name, r in results.items()},
        )

    # ------------------------------------------------------------------ logging

    def _embed_stats(self) -> tuple[int, float] | None:
        misses, seconds = getattr(self.embedder, "misses", None), getattr(self.embedder, "embed_seconds", None)
        return None if misses is None or seconds is None else (misses, seconds)

    def _log_ready(self, seconds: float) -> None:
        if not log.isEnabledFor(logging.INFO):
            return
        parts = [f"{len(self.corpus)} submissions", f"embedder {self.embedder.name}"]
        whole = next((s for s in self.signals if s.name == "whole_text"), None)
        rel = next((s for s in self.signals if s.name == "relevance"), None)
        if whole is not None and hasattr(whole, "scale"):
            parts.append(f"novelty median {whole.scale.median:.3f} (scale {whole.scale.scale:.3f})")
        if rel is not None and hasattr(rel, "boundary") and not getattr(rel, "uses_fallback", False):
            parts.append(f"learned relevance from {len(self.index.negative_content)} hard negatives")
        elif rel is not None and hasattr(rel, "on_topic_margin"):
            parts.append(f"typical relevance margin {rel.on_topic_margin:+.3f}")
        log.info("scorer ready: %s in %.0f ms", ", ".join(parts), seconds * 1000)

    @staticmethod
    def _log_input(action: str, sub: Submission) -> None:
        log.info('%s input: id=%s stance=%s words=%d headline="%s"', action, sub.id or "-",
                 sub.stance.value, len(sub.body.split()), preview(sub.headline))
        log.debug('%s body: "%s"', action, printable(sub.body))  # escaped: a newline must not forge a log line

    def _log_result(self, action: str, sub: Submission, r: ScoreBreakdown, start: float,
                    stats: tuple[int, float] | None, admission: str | None = None) -> None:
        if not log.isEnabledFor(logging.INFO):
            return
        total_ms = (time.perf_counter() - start) * 1000
        clause = "-" if r.clause_novelty is None else f"{r.clause_novelty:.2f}"
        margin = "-" if r.relevance_margin is None else f"{r.relevance_margin:+.3f}"
        parts = [f"score={r.score:.3f} = novelty {r.novelty:.3f} x gate {r.relevance_gate:.2f}",
                 f"whole {r.semantic_novelty:.2f} clause {clause}",
                 f"relevance {r.relevance:.2f} margin {margin}"]
        if admission is not None:
            parts.append(admission)
        timing = f"{total_ms:.0f} ms"
        now = self._embed_stats()
        if stats is not None and now is not None:
            timing += f" (embed {(now[1] - stats[1]) * 1000:.0f} ms, {now[0] - stats[0]} new)"
        parts.append(timing)
        flags = [x for x in r.reasons if not x.startswith(WholeTextNovelty.Z_REASON_PREFIX) and x != admission]
        if flags:
            parts.append("flags: " + preview("; ".join(flags), 160))
        log.info("%s result: id=%s %s", action, sub.id or "-", " | ".join(parts))

    def _log_calculation(self, r: ScoreBreakdown, a: Analysis) -> None:
        signals = " ".join(f"{name}={value:.3f}" for name, value in r.signals.items())
        nearest = " ".join(f"{n.id}:{n.similarity:.2f}" for n in r.nearest)
        prep = a.prepared
        fixes = ", ".join(f"{old}->{new or '(removed)'}" for old, new in prep.corrections) or "none"
        log.debug("calculation: %s | nearest %s | clauses %d (substantive %d, foreign %d) | corrections: %s",
                  signals, nearest, len(prep.clauses), sum(prep.substantive), sum(prep.foreign), preview(fixes, 120))

    # ------------------------------------------------------------------ inspection

    def corpus_novelty(self) -> dict[str, float]:
        """Leave-one-out whole-text novelty of each corpus item."""
        sig = self.signal("whole_text")
        return {k: round(sig.scale(v), 4) for k, v in sig.loo.items()}

    def corpus_relevance(self) -> dict[str, float]:
        """Leave-one-out calibrated relevance of each on-topic corpus item."""
        return self.signal("relevance").corpus_relevance()
