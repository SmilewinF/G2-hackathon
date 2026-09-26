"""End-to-end behaviour against the whole corpus (seed + admitted web-UI submissions), using a
real embedding model.

Probe texts live in data/probes.json and are disjoint from the seed corpus and calibration
anchors. If a probe was submitted through the web UI, the tests expect it to be caught as a
copy instead of rewarded (see helpers.assert_novel).
"""

import pytest

from helpers import NOT_NOVEL_MAX, NOVEL_MIN, UNREWARDED, USER_IDS, already_submitted, assert_novel, corpus_entry
from novelty.data import load_probes
from novelty.data import load_probes
from novelty.models import Stance, Submission

PROBE_COUNTS = {group: len(items) for group, items in load_probes().items()}  # parametrize over every probe

# ---------------------------------------------------------------- the four required behaviours


@pytest.mark.parametrize("idx", range(PROBE_COUNTS["novel_relevant"]))
def test_truly_novel_relevant_content_is_rewarded(scorer, probes, idx):
    sub = probes["novel_relevant"][idx]
    r = assert_novel(scorer, sub)
    assert r.relevance_gate == 1.0


@pytest.mark.parametrize("idx", range(PROBE_COUNTS["duplicates"]))
def test_non_novel_content_is_not_rewarded(scorer, probes, idx):
    sub = probes["duplicates"][idx]
    r = scorer.score(sub)
    assert r.score <= NOT_NOVEL_MAX, (sub.id, r)
    assert r.relevance_gate > 0.9, "these are on-topic: the low score must come from novelty"


@pytest.mark.parametrize("idx", range(PROBE_COUNTS["off_topic"]))
def test_high_novelty_but_low_relevance_is_not_rewarded(scorer, probes, idx):
    sub = probes["off_topic"][idx]
    r = scorer.score(sub)
    assert r.semantic_novelty >= 0.8, "off-topic text is maximally unlike the corpus, i.e. highly 'novel'"
    assert r.relevance_margin <= 0.0
    assert r.score <= UNREWARDED, (sub.id, r)


ADJACENT = load_probes()["off_topic_adjacent"]


@pytest.mark.parametrize("probe", ADJACENT, ids=lambda p: p["id"])
def test_same_town_civic_comments_about_other_subjects_are_not_rewarded(scorer, probe):
    """Bus routes, library hours, water rates, snow plowing, polling places, school start times:
    same town and civic tone, but not a response to this article. The contrastive-margin gate let
    these through at 0.45-0.97; the learned relevance gate must give them nothing."""
    r = scorer.score(Submission.from_dict(probe))
    assert r.score <= UNREWARDED, (probe["id"], r.score, r.relevance_gate)


def test_same_town_different_subject_is_not_rewarded(scorer, probes):
    """The hard case: shares the locale and comment genre but not the subject."""
    football = next(p for p in probes["off_topic"] if p.id == "o_football")
    r = scorer.score(football)
    assert r.semantic_novelty >= 0.8
    assert r.score <= UNREWARDED


def test_every_novel_submission_outranks_every_other_probe(scorer, probes):
    fresh = [s for s in probes["novel_relevant"] if not already_submitted(scorer, s)]
    if not fresh:
        pytest.skip("every novel probe has already been submitted through the web UI")
    novel = [scorer.score(s).score for s in fresh]
    others = [scorer.score(s).score for g in ("duplicates", "off_topic") for s in probes[g]]
    assert min(novel) > max(others) + 0.3


# ---------------------------------------------------------------- specific non-novelty tactics


def test_verbatim_copy_with_light_edits_scores_zero(scorer, probes):
    copy = next(p for p in probes["duplicates"] if p.id == "d_copy")
    r = scorer.score(copy)
    assert r.near_duplicate_of in {"c01", *USER_IDS} and r.score == 0.0


def test_changing_only_the_stance_does_not_make_a_copy_novel(scorer, probes):
    flipped = next(p for p in probes["duplicates"] if p.id == "d_stance_flip")
    assert flipped.stance is Stance.UNDECIDED  # the rarest stance in the seed corpus
    r = scorer.score(flipped)
    assert r.stance_rarity > 0.5
    assert r.score == 0.0


def test_padding_a_copy_with_extra_words_is_still_a_copy(scorer):
    original = corpus_entry(scorer, "c27")
    padded = Submission(
        headline=original.headline + "!!",
        body="Honestly I have to say this. " + original.body + " Just my two cents.",
        stance=Stance.SUPPORT,
    )
    assert scorer.score(padded).near_duplicate_of in {original.id, *USER_IDS}


# ---------------------------------------------------------------- relative to the corpus over time


def test_novelty_is_relative_to_what_has_been_submitted(scorer, probes):
    """Once a new idea is in the corpus, the next person to submit it is no longer novel."""
    first = probes["novel_relevant"][0]  # stormwater / flooding idea
    second = Submission(
        headline="Make the new park a stormwater sponge",
        body="Rain gardens and an underground storage tank in the park would stop Elm Street basements "
        "flooding every spring and ease the pressure on the storm drains and sewer.",
        stance=Stance.SUPPORT,
    )
    if already_submitted(scorer, first):
        pytest.skip("the stormwater probe was already submitted through the web UI")
    before = scorer.score(second).score
    assert scorer.submit(first).score >= NOVEL_MIN
    after = scorer.score(second).score
    assert before >= NOVEL_MIN
    assert after <= NOT_NOVEL_MAX


def test_off_topic_spam_does_not_redefine_the_topic(scorer, probes, corpus_size):
    for i in range(20):
        spam = Submission(
            headline=f"Best sourdough tip #{i}",
            body=f"Feed your starter rye flour and keep it at {20 + i} degrees for a better rise and crumb.",
            stance=Stance.SUPPORT,
        )
        result = scorer.submit(spam)
        assert result.score <= UNREWARDED and result.admitted is False
    assert len(scorer.corpus) == corpus_size, "off-topic spam must not enter the reference corpus"
    for sub in probes["novel_relevant"]:
        assert_novel(scorer, sub)
    for sub in probes["off_topic"]:
        assert scorer.score(sub).score <= UNREWARDED


def test_crowded_takes_are_less_novel_than_one_off_takes_within_the_corpus(scorer):
    loo = scorer.corpus_novelty()
    crowded = ["c04", "c09", "c10", "c20"]  # "parking kills Main Street", "too expensive"
    one_off = ["c49", "c50", "c22"]  # rushed vote, farmers market, cost overruns
    assert max(loo[c] for c in crowded) < min(loo[c] for c in one_off)


def test_relevance_gate_keeps_nearly_all_genuine_responses(scorer):
    """The gate must not be so strict that ordinary on-topic comments lose their reward.

    Leave-one-out: each corpus comment is judged by a gate trained without it. With the learned
    gate, 94% keep a full gate; two seed comments would be blocked as new submissions: c36 (a
    short question about the shuttle) and c49 (a process complaint about the vote). That is the
    measured cost of blocking same-town off-topic text, and is documented in the README."""
    relevance = scorer.corpus_relevance()
    full = [c for c, r in relevance.items() if r >= 1.0]
    assert len(full) / len(relevance) >= 0.9
    blocked = [c for c, r in relevance.items() if c.startswith("c") and r == 0.0]
    assert len(blocked) <= 2, blocked


def test_all_scores_are_normalised(scorer, probes):
    for group in probes.values():
        for sub in group:
            r = scorer.score(sub)
            assert 0.0 <= r.score <= 1.0


def test_every_corpus_entry_resubmitted_is_caught_as_a_copy(scorer):
    """Holds for the whole corpus, including anything added through the web UI."""
    for sub in scorer.corpus:
        r = scorer.score(Submission(headline=sub.headline, body=sub.body, stance=sub.stance))
        assert r.near_duplicate_of is not None and r.score == 0.0, sub.id
