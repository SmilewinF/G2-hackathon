"""Model-free checks of the scoring primitives and the pipeline's structural guarantees."""

import copy
import logging
import math

import numpy as np
import pytest

from helpers import TOY_ANCHORS, HashEmbedder, toy_corpus

from novelty.data import load_fixed_content
from novelty.models import Stance, Submission
from novelty.scorer import NoveltyScorer, ScorerConfig, default_signals
from novelty.signals import CalibrationError, Kind, RobustScale, Signal, SignalResult, WholeTextNovelty, normal_cdf, smoothstep
from novelty.text import clause_spans, clauses, containment, fold, is_generic, is_substantive, normalize, shingles


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


def test_normalize_removes_every_format_character_and_lone_surrogates():
    # bidi isolates, a tag character and a lone surrogate were kept; each split a word in two
    assert normalize("pa\u2066rk\u2069 pl\U000e0041an\ud800 \u034fnow") == "park plan now"
    assert normalize("café") == "café", "accents are display text; only matching folds them"
    assert fold("Thė gȧragė") == "the garage"


def test_clause_spans_locate_every_clause():
    text = "Parking loss hurts shops, the park helps families. Trees cool it down, ok, and so on."
    spans = clause_spans(text)
    assert [c for c, _, _ in spans] == clauses(text)
    for clause, start, end in spans:
        assert text[start:end].split()[0] == clause.split()[0] and text[start:end].split()[-1] == clause.split()[-1]


@pytest.mark.parametrize("word, generic", [("loved", True), ("agreed", True), ("ideas", True), ("loving", True),
                                           ("greatly", True), ("fines", False), ("garage", False), ("lovely", True)])
def test_inflections_of_generic_words_are_generic(word, generic):
    assert is_generic(word) is generic


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


@pytest.fixture
def toy_scorer():
    return NoveltyScorer(load_fixed_content(), toy_corpus(), HashEmbedder(), TOY_ANCHORS)


def test_every_score_component_is_bounded(toy_scorer):
    for text in ["the garage park downtown council plan", "the football goal match striker", "a zebra plays the violin"]:
        r = toy_scorer.score(Submission(headline="x", body=text + " is what I think about it", stance="mixed"))
        for value in (r.score, r.novelty, r.semantic_novelty, r.clause_novelty, r.stance_rarity, r.relevance, r.relevance_gate):
            if value is None:  # clause_novelty is None when exactly one clause is substantive
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
        NoveltyScorer(load_fixed_content(), toy_corpus()[:3], HashEmbedder(), TOY_ANCHORS)


class ConstantEmbedder:
    """A degenerate backend: every text maps to the same vector."""

    name = "constant"

    def embed(self, texts):
        return np.ones((len(texts), 8), dtype=np.float32) / np.sqrt(8)


def test_calibration_fails_loudly_when_the_embedder_cannot_separate_topic_from_chatter():
    with pytest.raises(CalibrationError, match="not distinguishable"):
        NoveltyScorer(load_fixed_content(), toy_corpus(), ConstantEmbedder(), TOY_ANCHORS)


def test_duplicate_ids_are_rejected(toy_scorer):
    with pytest.raises(ValueError, match="duplicate submission id"):
        toy_scorer.add(toy_scorer.corpus[0])


def test_signal_names_must_be_unique():
    with pytest.raises(ValueError, match="unique"):
        NoveltyScorer(load_fixed_content(), toy_corpus(), HashEmbedder(), TOY_ANCHORS,
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
    scorer = NoveltyScorer(load_fixed_content(), toy_corpus(), HashEmbedder(), TOY_ANCHORS,
                           signals=[*default_signals(toy_scorer.config), BannedWord()])
    sub = Submission(headline="Deal", body="The garage park plan is a scam and the council knows it", stance="oppose")
    r = scorer.score(sub)
    assert r.signals["banned_word"] == 0.0 and r.score == 0.0
    assert "contains a banned word" in r.reasons


def test_signal_results_outside_unit_interval_are_rejected():
    with pytest.raises(ValueError):
        SignalResult(1.5)


# ---------------------------------------------------------------- incremental calibration


def _assert_same_state(a, b, where):
    """Every calibrated attribute of two signal objects is equal (arrays and floats to 1e-9)."""
    if isinstance(a, np.ndarray):
        assert a.shape == b.shape and np.allclose(a, b, rtol=0, atol=1e-9, equal_nan=True), where
    elif isinstance(a, float):
        assert a == pytest.approx(b, abs=1e-9), where
    elif isinstance(a, dict):
        assert a.keys() == b.keys(), where
        for k in a:
            _assert_same_state(a[k], b[k], f"{where}[{k!r}]")
    elif isinstance(a, RobustScale):
        _assert_same_state(a.median, b.median, f"{where}.median")
        _assert_same_state(a.scale, b.scale, f"{where}.scale")
    elif isinstance(a, Signal):
        assert vars(a).keys() == vars(b).keys(), where
        for k, v in vars(a).items():
            if k != "_index":  # the index itself: shared by both copies
                _assert_same_state(v, vars(b)[k], f"{where}.{k}")
    else:
        assert a == b, where


@pytest.mark.parametrize("learned_gate", [False, True], ids=["margin gate", "learned gate"])
def test_incremental_calibration_matches_a_full_refit(caplog, learned_gate):
    """update() must reach exactly the state fit() would compute on the same index, for every
    signal, after single adds, a TF-IDF refit and a batch add. A signal whose update() raises
    silently falls back to fit(), so that fallback must not happen either."""
    scorer = NoveltyScorer(load_fixed_content(), toy_corpus(), HashEmbedder(), TOY_ANCHORS,
                           relevance_negatives=_toy_negatives() if learned_gate else ())
    assert scorer.signal("relevance").uses_fallback is not learned_gate
    extra = [Submission(headline=f"Extra {i}", body=f"The garage could host market stall number {i} "
                        f"with the downtown council support every weekend.", stance="mixed", id=f"e{i}")
             for i in range(10)]
    # 13 entries at construction; TF-IDF refits (a full fit of the novelty signals) at 15, 17, 19,
    # 21 and 24 entries. Eight single adds, then two in one batch (21 -> 23, as the server's replay
    # does), so the state is checked after incremental steps of both kinds, not only after refits.
    steps = [[sub] for sub in extra[:8]] + [extra[8:]]
    with caplog.at_level(logging.WARNING, logger="novelty.scorer"):
        for batch in steps:
            version = scorer.index.version
            scorer.add_many(batch)
            incremental_step = scorer.index.version == version
            for s in scorer.signals:
                full = copy.deepcopy(s)
                full.fit(scorer.index)
                _assert_same_state(s, full, f"{s.name} after {len(scorer.index)} entries")
            assert scorer.index.version == version or not incremental_step, "fit() must not refit the TF-IDF"
    assert incremental_step, "the batch add must be an incremental step"
    assert "failed to update incrementally" not in caplog.text


def test_tfidf_refits_periodically_not_on_every_insert():
    scorer = NoveltyScorer(load_fixed_content(), toy_corpus(), HashEmbedder(), TOY_ANCHORS)
    v0 = scorer.index.version
    versions = []
    for i in range(10):
        scorer.add(Submission(headline=f"More {i}", body=f"The council plan for the garage is idea {i} about "
                              f"the downtown park.", stance="support", id=f"m{i}"))
        versions.append(scorer.index.version)
    refits = len(set(versions) - {v0})
    assert 1 <= refits <= 5, versions  # 13 -> 23 entries in 10% growth steps: refits at 15, 17, 19, 21


def test_fork_is_independent_of_its_source(toy_scorer):
    fork = toy_scorer.fork()
    fork.add(Submission(headline="Fork only", body="The garage could become a covered skate park for the downtown kids.",
                        stance="support", id="f1"))
    assert len(fork.corpus) == len(toy_scorer.corpus) + 1
    assert toy_scorer.index.get("f1") is None
    assert fork.embedder is toy_scorer.embedder  # shared, not copied


# ---------------------------------------------------------------- learned relevance


def _toy_negatives(n=12):
    return [Submission(headline=f"Bus route {i}", body=f"The number {i} bus is late every morning and the drivers are short staffed.",
                       stance="oppose", id=f"neg{i}") for i in range(n)]


def test_learned_relevance_falls_back_to_the_margin_without_enough_negatives(toy_scorer):
    from novelty.signals import TopicMargin

    rel = toy_scorer.signal("relevance")
    assert rel.uses_fallback
    explicit = NoveltyScorer(load_fixed_content(), toy_corpus(), HashEmbedder(), TOY_ANCHORS,
                             signals=[*(s for s in default_signals(ScorerConfig()) if s.name != "relevance"), TopicMargin()])
    sub = Submission(headline="Flood park", body="The park and garage site could hold a flood basin for downtown.", stance="mixed")
    assert toy_scorer.score(sub).score == explicit.score(sub).score


def test_learned_relevance_is_used_when_negatives_are_given():
    scorer = NoveltyScorer(load_fixed_content(), toy_corpus(), HashEmbedder(), TOY_ANCHORS,
                           relevance_negatives=_toy_negatives())
    rel = scorer.signal("relevance")
    assert not rel.uses_fallback and len(scorer.index.negative_content) == 12
    r = scorer.score(Submission(headline="Buses", body="The number 7 bus is late every morning and short staffed.", stance="oppose"))
    assert 0.0 <= r.relevance_gate <= 1.0 and r.relevance_margin is not None
    assert all(0.0 <= v <= 1.0 for v in scorer.corpus_relevance().values())
