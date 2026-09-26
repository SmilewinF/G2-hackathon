"""Shared test helpers: thresholds, the test-corpus mode, novelty assertions, and a model-free toy embedder and corpus.

Tests run against the WHOLE corpus by default: the 50 seed comments in data/corpus.json plus
every admitted submission the web UI saved to data/user_submissions.json. Set
NOVELTY_TEST_CORPUS=seed to pin the suite to the seed corpus (reproducible CI runs).
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
import zlib

import numpy as np
import pytest

from novelty.data import load_user_submissions
from novelty.models import Stance, Submission

NOVEL_MIN = 0.6  # truly novel, relevant submissions must earn at least this
NOT_NOVEL_MAX = 0.25  # copies and paraphrases of existing takes must earn at most this
UNREWARDED = 0.01  # off-topic submissions earn (effectively) nothing

TEST_CORPUS = os.environ.get("NOVELTY_TEST_CORPUS", "full").lower()
if TEST_CORPUS not in ("full", "seed"):
    raise RuntimeError(f"NOVELTY_TEST_CORPUS must be 'full' or 'seed', got {TEST_CORPUS!r}")
USER_SUBMISSIONS = load_user_submissions() if TEST_CORPUS == "full" else []
USER_IDS = frozenset(s.id for s in USER_SUBMISSIONS)


def assert_novel(scorer, sub, minimum: float = NOVEL_MIN):
    """Assert a submission that is novel against the seed corpus is rewarded.

    Against the whole corpus the expectation changes when a web-UI submission already made the
    same point: an exact copy must earn nothing; an idea whose nearest neighbour is a web-UI
    submission and that no longer clears ``minimum`` is skipped (reported, naming the entry),
    because "novel" was defined relative to the seed corpus.
    """
    r = scorer.score(sub)
    if r.near_duplicate_of in USER_IDS:
        assert r.score == 0.0, (sub.id, r)
        return r
    if r.score < minimum and r.nearest and r.nearest[0].id in USER_IDS:
        pytest.skip(f"idea already covered by web-UI submission {r.nearest[0].id} (score {r.score})")
    assert r.score >= minimum, f"{sub.id or sub.headline!r} scored {r.score} < {minimum}; nearest {r.nearest[:3]}"
    return r


def already_submitted(scorer, sub) -> str | None:
    """Id of a web-UI submission that this text copies, if any."""
    r = scorer.score(sub)
    return r.near_duplicate_of if r.near_duplicate_of in USER_IDS else None


def corpus_entry(scorer, entry_id: str):
    """The corpus submission with this id."""
    return next(s for s in scorer.corpus if s.id == entry_id)


def request_json(url: str, body: bytes | None = None) -> tuple[int, dict]:
    """(status, JSON payload) for a GET (no body) or a JSON POST, including error responses."""
    req = urllib.request.Request(url, body, {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req) as resp:
            return resp.status, json.load(resp)
    except urllib.error.HTTPError as e:
        return e.code, json.load(e)


class HashEmbedder:
    """Deterministic bag-of-words embedder so scorer invariants can be tested without a model."""

    name = "hash"

    def embed(self, texts):
        out = np.zeros((len(texts), 64), dtype=np.float32)
        for i, t in enumerate(texts):
            for w in t.lower().split():
                out[i, zlib.crc32(w.encode()) % 64] += 1.0
        return out / np.clip(np.linalg.norm(out, axis=1, keepdims=True), 1e-12, None)


TOY_ANCHORS = ["the football match was great and the striker scored twice in the final minutes"]


def toy_corpus() -> list[Submission]:
    """Twelve near-identical on-topic comments (every fourth opposes) for model-free scorer tests."""
    return [
        Submission(
            headline=f"Garage idea {i}",
            body=f"The council plan for the garage and the park is idea number {i} for the downtown area.",
            stance=Stance.SUPPORT if i % 4 else Stance.OPPOSE,
        )
        for i in range(12)
    ]
