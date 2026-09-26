"""Model-free checks of the scoring primitives."""

import math
import zlib

import numpy as np
import pytest

from novelty.lexical import containment, shingles
from novelty.models import Stance, Submission
from novelty.scorer import NoveltyScorer, _normal_cdf, _smoothstep
from novelty.data import load_fixed_content


def test_smoothstep_is_zero_below_floor_and_one_above_full():
    assert _smoothstep(0.1, 0.5, 0.0) == 0.0
    assert _smoothstep(0.1, 0.5, 0.1) == 0.0
    assert _smoothstep(0.1, 0.5, 0.5) == 1.0
    assert 0.0 < _smoothstep(0.1, 0.5, 0.3) < 1.0


def test_normal_cdf_maps_z_scores_into_unit_interval():
    assert _normal_cdf(0.0) == pytest.approx(0.5)
    assert _normal_cdf(-5) < 1e-6 and _normal_cdf(5) > 1 - 1e-6


def test_containment_detects_copies_and_padded_copies():
    a = shingles("Take away 600 spaces and customers will go to the mall.")
    padded = shingles("Honestly, take away 600 spaces and customers will go to the mall. Sad!")
    other = shingles("A shady park downtown could save lives during heat waves.")
    assert containment(a, a) == 1.0
    assert containment(a, padded) > 0.9
    assert containment(a, other) < 0.3


class HashEmbedder:
    """Deterministic bag-of-words embedder so scorer invariants can be tested without a model."""

    name = "hash"

    def embed(self, texts):
        out = np.zeros((len(texts), 64), dtype=np.float32)
        for i, t in enumerate(texts):
            for w in t.lower().split():
                out[i, zlib.crc32(w.encode()) % 64] += 1.0
        return out / np.clip(np.linalg.norm(out, axis=1, keepdims=True), 1e-12, None)


def _sub(i, stance=Stance.SUPPORT):
    return Submission(headline=f"garage park idea {i}", body=f"parking garage park downtown council {i} " * 3, stance=stance)


@pytest.fixture
def toy_scorer():
    corpus = [_sub(i, Stance.SUPPORT if i % 4 else Stance.OPPOSE) for i in range(12)]
    return NoveltyScorer(load_fixed_content(), corpus, HashEmbedder(), ["football goal match striker keeper"])


def test_every_score_component_is_bounded(toy_scorer):
    for text in ["garage park downtown council", "football goal match striker", "zebra quantum violin"]:
        r = toy_scorer.score(Submission(headline="x", body=text + " filler text here", stance="mixed"))
        for value in (r.score, r.novelty, r.semantic_novelty, r.stance_rarity, r.relevance, r.relevance_gate):
            assert 0.0 <= value <= 1.0 and not math.isnan(value)


def test_score_is_novelty_times_gate(toy_scorer):
    r = toy_scorer.score(Submission(headline="new", body="park garage flood basin downtown", stance="mixed"))
    assert r.score == pytest.approx(r.novelty * r.relevance_gate, abs=1e-3)


def test_most_common_stance_has_zero_rarity(toy_scorer):
    assert toy_scorer._stance_rarity(Stance.SUPPORT) == 0.0
    assert toy_scorer._stance_rarity(Stance.UNDECIDED) > toy_scorer._stance_rarity(Stance.OPPOSE)


def test_corpus_must_exceed_neighbourhood_size():
    with pytest.raises(ValueError):
        NoveltyScorer(load_fixed_content(), [_sub(i) for i in range(3)], HashEmbedder(), ["x y z"])
