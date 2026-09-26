"""The web UI's examples (data/examples.json) must demonstrate what their labels claim."""

import pytest

from helpers import NOT_NOVEL_MAX, UNREWARDED, assert_novel
from novelty.data import load_corpus, load_examples, load_probes
from novelty.models import Stance, Submission

EXAMPLES = [e for items in load_examples().values() for e in items]


def test_two_examples_per_stance():
    examples = load_examples()
    assert list(examples) == [s.value for s in Stance]
    assert all(len(items) == 2 for items in examples.values())


def test_examples_are_unique_and_not_test_fixtures():
    ids = [e["id"] for e in EXAMPLES]
    assert len(set(ids)) == len(ids)
    seen = {s.body for s in load_corpus()} | {p["body"] for g in load_probes().values() for p in g}
    assert not any(e["body"] in seen for e in EXAMPLES)


@pytest.mark.parametrize("example", EXAMPLES, ids=lambda e: e["id"])
def test_example_behaves_as_labelled(scorer, example):
    sub = Submission.from_dict(example)
    if example["expect"] == "novel":
        assert_novel(scorer, sub)
    elif example["expect"] == "common":
        assert scorer.score(sub).score <= NOT_NOVEL_MAX
    else:
        assert scorer.score(sub).score <= UNREWARDED
