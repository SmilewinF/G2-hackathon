"""The held-out evaluation set (data/eval/heldout.json): integrity, and the aggregate behaviour
the brief asks us to demonstrate, measured on the TEST split, which no threshold was tuned on.

The pass-rate floors for new ideas and paraphrases are regression guards set about 10 points
below the measured values (see README "Held-out evaluation"), not claims of quality. The
off-topic and relevance bars are the must-show property of the brief.
"""

import json

import pytest

from novelty.data import DATA_DIR, build_scorer, load_corpus, load_examples, load_probes, load_relevance_negatives
from novelty.evaluation import EVAL_FILE, THRESHOLDS, evaluate, load_eval_set

RAW = json.loads(EVAL_FILE.read_text(encoding="utf-8"))["items"]


# ---------------------------------------------------------------- integrity


def test_eval_set_has_dev_and_test_splits_with_every_label():
    splits = {d["split"] for d in RAW}
    assert splits == {"dev", "test"}
    for split in splits:
        labels = {d["label"] for d in RAW if d["split"] == split}
        assert {"novel", "paraphrase", "generic", "off_topic", "seq_first", "seq_second"} <= labels


def test_sequential_pairs_stay_in_one_split():
    pairs = {}
    for d in RAW:
        if d["label"].startswith("seq_"):
            pairs.setdefault(d["of"], set()).add(d["split"])
    assert pairs and all(len(s) == 1 for s in pairs.values())


def test_eval_items_are_disjoint_from_everything_the_scorer_learns_from():
    bodies = {d["body"] for d in RAW}
    assert len(bodies) == len(RAW), "duplicate evaluation items"
    seen = {s.body for s in load_corpus()} | {s.body for s in load_relevance_negatives()}
    seen |= {p["body"] for g in load_probes().values() for p in g}
    seen |= {e["body"] for g in load_examples().values() for e in g}
    assert not bodies & seen


def test_every_item_has_an_expectation():
    for item in load_eval_set(split="all"):
        assert item.label in THRESHOLDS or item.label == "seq_second" or item.expect is not None, item.id


def test_relevance_negatives_are_a_separate_training_set():
    negatives = load_relevance_negatives()
    assert len(negatives) >= 50
    assert (DATA_DIR / "relevance_negatives.json").exists()


# ---------------------------------------------------------------- aggregate behaviour (test split)


@pytest.fixture(scope="module")
def reports(embedder):
    items = load_eval_set(split="test")
    return {
        "learned": evaluate(lambda: build_scorer(embedder), items, "test"),
        "margin_only": evaluate(lambda: build_scorer(embedder, use_relevance_negatives=False), items, "test"),
    }


def test_off_topic_content_is_not_rewarded_on_unseen_inputs(reports):
    """The brief's must-show property, on held-out inputs: high novelty, low relevance earns nothing."""
    r = reports["learned"]
    assert r.pass_rates["off_topic"] >= 0.85, r.failures
    assert r.auc_relevance >= 0.85


def test_learned_relevance_beats_the_contrastive_margin_on_off_topic(reports):
    before, after = reports["margin_only"].pass_rates["off_topic"], reports["learned"].pass_rates["off_topic"]
    assert after - before >= 0.4, (before, after)


def test_off_topic_topics_absent_from_the_training_negatives_are_still_blocked(reports):
    """Generalisation, not memorisation: the negatives never covered these subjects."""
    assert reports["learned"].by_topic["adjacent_unseen"] >= 0.8


def test_novelty_regression_floors(reports):
    r = reports["learned"]
    assert r.pass_rates["novel"] >= 0.45, r.failures
    assert r.pass_rates["paraphrase"] >= 0.40
    assert r.auc_novelty >= 0.70
