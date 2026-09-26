"""The content shape required by the brief."""

import dataclasses

import pytest

from novelty.data import load_corpus, load_fixed_content, load_off_topic_anchors, load_probes
from novelty.models import FixedContent, Stance, Submission


def test_fixed_content_is_at_most_100_words():
    assert len(load_fixed_content().text.split()) <= 100


def test_fixed_content_over_100_words_is_rejected():
    with pytest.raises(ValueError):
        FixedContent(id="x", title="t", text="word " * 101)


def test_submission_has_three_user_provided_properties():
    user_fields = [f.name for f in dataclasses.fields(Submission) if f.name != "id"]
    assert user_fields == ["headline", "body", "stance"]


def test_stance_is_multiple_choice():
    assert Submission(headline="h", body="x" * 30, stance="support").stance is Stance.SUPPORT
    with pytest.raises(ValueError):
        Submission(headline="h", body="x" * 30, stance="strongly agree")


@pytest.mark.parametrize(
    "headline, body",
    [("", "x" * 30), ("h" * 121, "x" * 30), ("ok", "too short")],
)
def test_invalid_submissions_are_rejected(headline, body):
    with pytest.raises(ValueError):
        Submission(headline=headline, body=body, stance=Stance.MIXED)


def test_corpus_has_about_fifty_unique_submissions():
    corpus = load_corpus()
    assert 45 <= len(corpus) <= 55
    assert len({s.id for s in corpus}) == len(corpus)


def test_probes_and_anchors_are_not_in_the_corpus():
    """Novel and off-topic probes are unseen. (Duplicate probes are derived from the corpus on purpose.)"""
    corpus_text = {s.text for s in load_corpus()}
    anchors = set(load_off_topic_anchors())
    for group in ("novel_relevant", "off_topic", "off_topic_adjacent"):
        for p in load_probes()[group]:
            text = Submission.from_dict(p).text
            assert text not in corpus_text
            assert p["body"] not in anchors
