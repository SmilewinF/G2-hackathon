import os

import pytest

# Tests are pinned to the local model so they are deterministic, offline and key-free.
# Set NOVELTY_TEST_EMBEDDER=gemini to run the same behavioural suite against Gemini.
os.environ["NOVELTY_EMBEDDER"] = os.environ.get("NOVELTY_TEST_EMBEDDER", "local")

from novelty.data import build_scorer, load_probes  # noqa: E402
from novelty.embeddings import default_embedder  # noqa: E402
from novelty.models import Submission  # noqa: E402


@pytest.fixture(scope="session")
def embedder():
    return default_embedder()


@pytest.fixture(scope="session")
def base_scorer(embedder):
    return build_scorer(embedder)


@pytest.fixture
def scorer(base_scorer):
    """Independent scorer per test (``submit`` mutates the corpus), forked from one build."""
    return base_scorer.fork()


@pytest.fixture(scope="session")
def probes():
    return {group: [Submission.from_dict(p) for p in items] for group, items in load_probes().items()}
