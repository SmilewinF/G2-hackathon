import os
import threading
from http.server import ThreadingHTTPServer

import pytest

# Tests are pinned to the local model so they are deterministic, offline and key-free.
# Set NOVELTY_TEST_EMBEDDER=gemini to run the same behavioural suite against Gemini.
os.environ["NOVELTY_EMBEDDER"] = os.environ.get("NOVELTY_TEST_EMBEDDER", "local")

from helpers import TEST_CORPUS, USER_SUBMISSIONS  # noqa: E402
from novelty.data import build_scorer, load_probes  # noqa: E402
from novelty.embeddings import default_embedder  # noqa: E402
from novelty.models import Submission  # noqa: E402
from novelty.server import App, make_handler  # noqa: E402


def pytest_report_header(config):
    if TEST_CORPUS == "seed":
        return "novelty test corpus: 50 seed comments (set NOVELTY_TEST_CORPUS=full to add the web-UI submissions)"
    return (f"novelty test corpus: seed comments + {len(USER_SUBMISSIONS)} admitted web-UI submission(s) "
            f"from data/user_submissions.json (results depend on those submissions)")


@pytest.fixture(scope="session")
def embedder():
    emb = default_embedder()
    yield emb
    emb.close()


@pytest.fixture(scope="session")
def base_scorer(embedder):
    return build_scorer(embedder, include_user_submissions=TEST_CORPUS == "full")


@pytest.fixture(scope="session")
def corpus_size(base_scorer):
    """Size of the test corpus (seed + web-UI submissions in full mode)."""
    return len(base_scorer.corpus)


@pytest.fixture
def scorer(base_scorer):
    """Independent scorer per test (``submit`` mutates the corpus), forked from one build."""
    return base_scorer.fork()


@pytest.fixture(scope="session")
def probes():
    return {group: [Submission.from_dict(p) for p in items] for group, items in load_probes().items()}


@pytest.fixture
def user_file(tmp_path):
    return tmp_path / "user_submissions.json"


@pytest.fixture
def app(base_scorer, user_file):
    return App(base_scorer.fork, user_file)


@pytest.fixture
def base_url(app):
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(app))
    # short poll interval: shutdown() otherwise waits up to 0.5 s per test
    threading.Thread(target=httpd.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_port}"
    httpd.shutdown()
    httpd.server_close()  # the listening socket otherwise leaks (ResourceWarning)
