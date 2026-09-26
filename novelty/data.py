"""Loaders for the bundled fixed content, corpus, calibration anchors and test probes."""

from __future__ import annotations

import json
from pathlib import Path

from .embeddings import Embedder, default_embedder
from .models import FixedContent, Submission
from .scorer import NoveltyScorer, ScorerConfig

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def _load(name: str):
    return json.loads((DATA_DIR / name).read_text(encoding="utf-8"))


def load_fixed_content() -> FixedContent:
    return FixedContent(**_load("fixed_content.json"))


def load_corpus() -> list[Submission]:
    return [Submission.from_dict(d) for d in _load("corpus.json")]


def load_off_topic_anchors() -> list[str]:
    return _load("off_topic_anchors.json")


def load_probes() -> dict[str, list[dict]]:
    """Labelled probe submissions grouped by expected behaviour (see data/probes.json)."""
    return {k: v for k, v in _load("probes.json").items() if not k.startswith("_")}


def build_scorer(embedder: Embedder | None = None, config: ScorerConfig = ScorerConfig()) -> NoveltyScorer:
    return NoveltyScorer(
        fixed=load_fixed_content(),
        corpus=load_corpus(),
        embedder=embedder or default_embedder(),
        off_topic_anchors=load_off_topic_anchors(),
        config=config,
    )
