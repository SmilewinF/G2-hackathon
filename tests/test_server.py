"""The web UI's JSON API, exercised over a real socket."""

import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from novelty.data import build_scorer
from novelty.server import App, make_handler


@pytest.fixture
def user_file(tmp_path):
    return tmp_path / "user_submissions.json"


@pytest.fixture
def app(embedder, user_file):
    return App(lambda: build_scorer(embedder), user_file)


@pytest.fixture
def base_url(app):
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(app))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_port}"
    httpd.shutdown()


def _post(url, payload):
    req = urllib.request.Request(url, json.dumps(payload).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as resp:
        return json.load(resp)


def test_page_and_context_are_served(base_url):
    with urllib.request.urlopen(base_url + "/") as resp:
        assert b"Novelty Scorer" in resp.read()
    with urllib.request.urlopen(base_url + "/api/context") as resp:
        ctx = json.load(resp)
    assert len(ctx["corpus"]) == 50 and ctx["stances"] and ctx["probes"]["novel_relevant"]


def test_score_does_not_change_corpus_but_commit_does(base_url, probes):
    p = probes["novel_relevant"][0]
    payload = {"headline": p.headline, "body": p.body, "stance": p.stance.value}
    scored = _post(base_url + "/api/score", payload)
    assert scored["result"]["score"] >= 0.6 and len(scored["corpus"]) == 50
    committed = _post(base_url + "/api/score", {**payload, "commit": True})
    assert committed["id"] == "u01" and len(committed["corpus"]) == 51
    again = _post(base_url + "/api/score", payload)
    assert again["result"]["near_duplicate_of"] == "u01" and again["result"]["score"] == 0.0


def test_invalid_submission_returns_400(base_url):
    with pytest.raises(urllib.error.HTTPError) as err:
        _post(base_url + "/api/score", {"headline": "", "body": "short", "stance": "maybe"})
    assert err.value.code == 400


def test_committed_submissions_are_saved_and_reloaded(base_url, probes, embedder, user_file):
    p = probes["novel_relevant"][1]
    _post(base_url + "/api/score", {"headline": p.headline, "body": p.body, "stance": p.stance.value, "commit": True})
    saved = json.loads(user_file.read_text(encoding="utf-8"))
    assert [r["id"] for r in saved] == ["u01"] and saved[0]["headline"] == p.headline and saved[0]["on_topic"]

    restarted = App(lambda: build_scorer(embedder), user_file)  # simulates a server restart
    assert len(restarted.scorer.corpus) == 51
    assert restarted.scorer.score(p).near_duplicate_of == "u01"


def test_off_topic_submissions_are_saved_but_not_part_of_the_topic(base_url, probes, embedder, user_file):
    p = probes["off_topic"][0]
    _post(base_url + "/api/score", {"headline": p.headline, "body": p.body, "stance": p.stance.value, "commit": True})
    assert json.loads(user_file.read_text(encoding="utf-8"))[0]["on_topic"] is False
    restarted = App(lambda: build_scorer(embedder), user_file)
    assert restarted.scorer._topic_member[-1] is False


def test_reset_forgets_user_submissions(base_url, probes, user_file):
    p = probes["novel_relevant"][2]
    _post(base_url + "/api/score", {"headline": p.headline, "body": p.body, "stance": p.stance.value, "commit": True})
    state = _post(base_url + "/api/reset", {})
    assert state["user_count"] == 0 and len(state["corpus"]) == 50 and not user_file.exists()
