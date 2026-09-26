"""The held-out evaluation set (data/eval/heldout.json): integrity, and the aggregate behaviour
the brief asks us to demonstrate.

The aggregate checks run on the DEV split by default. The same checks on the TEST split are
marked ``heldout_test`` and deselected (pyproject.toml): the test split is reported, never tuned
on, so an ordinary test run must not give pass/fail feedback on it. Run them for the report with
``pytest -m heldout_test`` (or ``python -m novelty eval --split test``).

The pass-rate floors are regression guards set about 10 points below the values measured on
each split (see README "Held-out evaluation"), not claims of quality. The off-topic and
relevance bars are the must-show property of the brief.
"""

import json
import re

import pytest

from novelty.data import (
    DATA_DIR,
    build_scorer,
    load_corpus,
    load_examples,
    load_fixed_content,
    load_off_topic_anchors,
    load_probes,
    load_relevance_negatives,
)
from novelty.evaluation import EVAL_FILE, THRESHOLDS, evaluate, load_eval_set
from novelty.text import fold, normalize

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


def _key(text: str) -> str:
    """Comparison form: normalised, accents folded, punctuation and spacing ignored."""
    return " ".join(re.sub(r"[^a-z0-9]+", " ", fold(normalize(text))).split())


def test_eval_items_are_disjoint_from_everything_the_scorer_learns_from():
    """Compared after normalisation, headlines as well as bodies: an exact-string check missed a
    copy that differed only in case, spacing or punctuation."""
    items = {d["split"]: set() for d in RAW}
    for d in RAW:
        items[d["split"]] |= {_key(d["body"]), _key(d["headline"])}
    bodies = [_key(d["body"]) for d in RAW]
    assert len(set(bodies)) == len(bodies), "duplicate evaluation items"
    learned = [*load_corpus(), *load_relevance_negatives()]
    shown = [p for g in load_probes().values() for p in g] + [e for g in load_examples().values() for e in g]
    seen = {_key(s.body) for s in learned} | {_key(s.headline) for s in learned}
    seen |= {_key(p["body"]) for p in shown} | {_key(p["headline"]) for p in shown}
    fixed = load_fixed_content()
    seen |= {_key(a) for a in load_off_topic_anchors()} | {_key(fixed.text), _key(fixed.title)}
    for split, keys in items.items():
        assert not keys & seen, (split, sorted(keys & seen)[:3])
    assert not {_key(d["body"]) for d in RAW if d["split"] == "dev"} & {_key(d["body"]) for d in RAW if d["split"] == "test"}


def test_every_item_has_an_expectation():
    for item in load_eval_set(split="all"):
        assert item.label in THRESHOLDS or item.label == "seq_second" or item.expect is not None, item.id


def test_relevance_negatives_are_a_separate_training_set():
    negatives = load_relevance_negatives()
    assert len(negatives) >= 50
    assert (DATA_DIR / "relevance_negatives.json").exists()


# ---------------------------------------------------------------- aggregate behaviour


FLOORS = {  # about 10 points below the values measured on each split
    "dev": {"off_topic": 0.85, "auc_relevance": 0.85, "gain": 0.3, "unseen": 0.8,
            "novel": 0.40, "paraphrase": 0.15, "auc_novelty": 0.60},
    "test": {"off_topic": 0.85, "auc_relevance": 0.85, "gain": 0.4, "unseen": 0.8,
             "novel": 0.45, "paraphrase": 0.40, "auc_novelty": 0.70},
}


@pytest.fixture(scope="module", params=["dev", pytest.param("test", marks=pytest.mark.heldout_test)])
def split(request):
    return request.param


@pytest.fixture(scope="module")
def reports(embedder, split):
    items = load_eval_set(split=split)
    return {
        "learned": evaluate(lambda: build_scorer(embedder), items, split),
        "margin_only": evaluate(lambda: build_scorer(embedder, use_relevance_negatives=False), items, split),
    }


def test_off_topic_content_is_not_rewarded_on_unseen_inputs(reports, split):
    """The brief's must-show property, on held-out inputs: high novelty, low relevance earns nothing."""
    r = reports["learned"]
    assert r.pass_rates["off_topic"] >= FLOORS[split]["off_topic"], r.failures
    assert r.auc_relevance >= FLOORS[split]["auc_relevance"]


def test_learned_relevance_beats_the_contrastive_margin_on_off_topic(reports, split):
    before, after = reports["margin_only"].pass_rates["off_topic"], reports["learned"].pass_rates["off_topic"]
    assert after - before >= FLOORS[split]["gain"], (before, after)


def test_off_topic_topics_absent_from_the_training_negatives_are_still_blocked(reports, split):
    """Generalisation, not memorisation: the negatives never covered these subjects."""
    assert reports["learned"].by_topic["adjacent_unseen"] >= FLOORS[split]["unseen"]


def test_novelty_regression_floors(reports, split):
    r, floor = reports["learned"], FLOORS[split]
    assert r.pass_rates["novel"] >= floor["novel"], r.failures
    assert r.pass_rates["paraphrase"] >= floor["paraphrase"]
    assert r.auc_novelty >= floor["auc_novelty"]
