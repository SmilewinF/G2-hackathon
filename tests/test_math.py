"""Model-free checks of the scoring primitives and the pipeline's structural guarantees."""

import math
import zlib

import numpy as np
import pytest

from novelty.data import load_fixed_content
from novelty.models import Stance, Submission
from novelty.scorer import NoveltyScorer
from novelty.signals import CalibrationError, Kind, RobustScale, Signal, SignalResult, normal_cdf, smoothstep
from novelty.text import clauses, containment, is_substantive, normalize, shingles


def test_smoothstep_is_zero_below_floor_and_one_above_full():
    assert smoothstep(0.1, 0.5, 0.0) == 0.0
    assert smoothstep(0.1, 0.5, 0.1) == 0.0
    assert smoothstep(0.1, 0.5, 0.5) == 1.0
    assert 0.0 < smoothstep(0.1, 0.5, 0.3) < 1.0


def test_normal_cdf_maps_z_scores_into_unit_interval():
    assert normal_cdf(0.0) == pytest.approx(0.5)
    assert normal_cdf(-5) < 1e-6 and normal_cdf(5) > 1 - 1e-6


def test_robust_scale_has_a_floor_for_constant_distributions():
    scale = RobustScale.fit(np.array([0.2] * 20))
    assert scale.scale > 0
    assert 0.0 < scale(0.21) < 1.0  # no division blow-up into exactly 0/1 for tiny differences


def test_containment_detects_copies_and_padded_copies():
    a = shingles("Take away 600 spaces and customers will go to the mall.")
    padded = shingles("Honestly, take away 600 spaces and customers will go to the mall. Sad!")
    other = shingles("A shady park downtown could save lives during heat waves.")
    assert containment(a, a) == 1.0
    assert containment(a, padded) > 0.9
    assert containment(a, other) < 0.3


def test_normalize_removes_invisible_characters_and_maps_lookalike_letters():
    assert normalize("pa​rk​ing") == "parking"
    assert normalize("gаrаgе") == "garage"  # Cyrillic а / е
    russian = "Парковка нужна городу"
    assert normalize(russian) == russian  # genuine non-Latin text is left alone


def test_clauses_split_lists_and_merge_fragments():
    parts = clauses("Parking loss hurts shops, the park helps families, and the vote was rushed.")
    assert parts == ["Parking loss hurts shops", "the park helps families", "the vote was rushed"]


@pytest.mark.parametrize(
    "clause, expected",
    [
        ("The council should have asked residents first", True),
        ("garage park Elm Street levy council shuttle parking downtown", False),  # keyword list
        ("park park park park park park park park", False),  # repetition
        ("ok", False),
    ],
)
def test_is_substantive(clause, expected):
    assert is_substantive(clause) is expected


class HashEmbedder:
    """Deterministic bag-of-words embedder so scorer invariants can be tested without a model."""

    name = "hash"

    def embed(self, texts):
        out = np.zeros((len(texts), 64), dtype=np.float32)
        for i, t in enumerate(texts):
            for w in t.lower().split():
                out[i, zlib.crc32(w.encode()) % 64] += 1.0
        return out / np.clip(np.linalg.norm(out, axis=1, keepdims=True), 1e-12, None)


ANCHORS = ["the football match was great and the striker scored twice in the final minutes"]


def _sub(i, stance=Stance.SUPPORT):
    return Submission(
        headline=f"Garage idea {i}",
        body=f"The council plan for the garage and the park is idea number {i} for the downtown area.",
        stance=stance,
    )


def _corpus():
    return [_sub(i, Stance.SUPPORT if i % 4 else Stance.OPPOSE) for i in range(12)]


@pytest.fixture
def toy_scorer():
    return NoveltyScorer(load_fixed_content(), _corpus(), HashEmbedder(), ANCHORS)


def test_every_score_component_is_bounded(toy_scorer):
    for text in ["the garage park downtown council plan", "the football goal match striker", "a zebra plays the violin"]:
        r = toy_scorer.score(Submission(headline="x", body=text + " is what I think about it", stance="mixed"))
        for value in (r.score, r.novelty, r.semantic_novelty, r.clause_novelty, r.stance_rarity, r.relevance, r.relevance_gate):
            if value is None:  # clause_novelty is None for single-clause text
                continue
            assert 0.0 <= value <= 1.0 and not math.isnan(value)


def test_score_is_novelty_times_gate(toy_scorer):
    r = toy_scorer.score(Submission(headline="new", body="the park and garage could hold a flood basin downtown", stance="mixed"))
    assert r.score == pytest.approx(r.novelty * r.relevance_gate, abs=1e-3)


def test_most_common_stance_has_zero_rarity(toy_scorer):
    stance = toy_scorer.signal("stance")
    assert stance.rarity(Stance.SUPPORT) == 0.0
    assert stance.rarity(Stance.UNDECIDED) > stance.rarity(Stance.OPPOSE)


def test_corpus_must_exceed_neighbourhood_size():
    with pytest.raises(ValueError):
        NoveltyScorer(load_fixed_content(), _corpus()[:3], HashEmbedder(), ANCHORS)


class ConstantEmbedder:
    """A degenerate backend: every text maps to the same vector."""

    name = "constant"

    def embed(self, texts):
        return np.ones((len(texts), 8), dtype=np.float32) / np.sqrt(8)


def test_calibration_fails_loudly_when_the_embedder_cannot_separate_topic_from_chatter():
    with pytest.raises(CalibrationError, match="not distinguishable"):
        NoveltyScorer(load_fixed_content(), _corpus(), ConstantEmbedder(), ANCHORS)


def test_duplicate_ids_are_rejected(toy_scorer):
    with pytest.raises(ValueError, match="duplicate submission id"):
        toy_scorer.add(toy_scorer.corpus[0])


def test_signal_names_must_be_unique():
    from novelty.signals import WholeTextNovelty

    with pytest.raises(ValueError, match="unique"):
        NoveltyScorer(load_fixed_content(), _corpus(), HashEmbedder(), ANCHORS,
                      signals=[WholeTextNovelty(), WholeTextNovelty()])


# ---------------------------------------------------------------- extensibility


class BannedWord(Signal):
    """Example plug-in: a moderation veto, added without touching the pipeline."""

    name = "banned_word"
    kind = Kind.MODIFIER

    def evaluate(self, analysis, index):
        hit = "scam" in analysis.text.lower()
        return SignalResult(0.0 if hit else 1.0, ["contains a banned word"] if hit else [])


def test_custom_signals_plug_into_the_pipeline(toy_scorer):
    from novelty.scorer import default_signals

    scorer = NoveltyScorer(load_fixed_content(), _corpus(), HashEmbedder(), ANCHORS,
                           signals=[*default_signals(toy_scorer.config), BannedWord()])
    sub = Submission(headline="Deal", body="The garage park plan is a scam and the council knows it", stance="oppose")
    r = scorer.score(sub)
    assert r.signals["banned_word"] == 0.0 and r.score == 0.0
    assert "contains a banned word" in r.reasons


def test_signal_results_outside_unit_interval_are_rejected():
    with pytest.raises(ValueError):
        SignalResult(1.5)


# ---------------------------------------------------------------- incremental calibration


def test_incremental_calibration_matches_a_full_refit():
    """update() must reach exactly the state fit() would compute on the same index."""
    import copy

    scorer = NoveltyScorer(load_fixed_content(), _corpus(), HashEmbedder(), ANCHORS)
    for i in range(3):  # stays below the TF-IDF refit threshold, so every update is incremental
        scorer.add(Submission(headline=f"Extra {i}", body=f"The garage could host market stall number {i} "
                              f"with the downtown council support every weekend.", stance="mixed", id=f"e{i}"))
    version = scorer.index.version
    for name in ("whole_text", "clause_coverage", "relevance"):
        incremental = scorer.signal(name)
        full = copy.copy(incremental)
        full.fit(scorer.index)
        for attr in ("scale", "loo", "on_topic_margin"):
            if hasattr(incremental, attr):
                a, b = getattr(incremental, attr), getattr(full, attr)
                if isinstance(a, dict):
                    assert a.keys() == b.keys()
                    assert all(a[k] == pytest.approx(b[k], abs=1e-9) for k in a), (name, attr)
                elif isinstance(a, float):
                    assert a == pytest.approx(b, abs=1e-9), (name, attr)
                else:
                    assert a.median == pytest.approx(b.median, abs=1e-9) and a.scale == pytest.approx(b.scale, abs=1e-9)
    assert scorer.index.version == version, "test assumption: no TF-IDF refit happened"


def test_tfidf_refits_periodically_not_on_every_insert():
    scorer = NoveltyScorer(load_fixed_content(), _corpus(), HashEmbedder(), ANCHORS)
    v0 = scorer.index.version
    versions = []
    for i in range(10):
        scorer.add(Submission(headline=f"More {i}", body=f"The council plan for the garage is idea {i} about "
                              f"the downtown park.", stance="support", id=f"m{i}"))
        versions.append(scorer.index.version)
    refits = len(set(versions) - {v0})
    assert 1 <= refits <= 5, versions  # 13 -> 23 entries in 10% growth steps: refits at 15, 17, 19, 21
