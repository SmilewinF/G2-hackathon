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
def app(base_scorer, user_file):
    return App(base_scorer.fork, user_file)


@pytest.fixture
def base_url(app):
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(app))
    # short poll interval: shutdown() otherwise waits up to 0.5 s per test
    threading.Thread(target=httpd.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True).start()
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
    assert ctx["corpus_size"] == 50 and ctx["stances"] and ctx["probes"]["novel_relevant"]


def test_score_does_not_change_corpus_but_commit_does(base_url, probes):
    p = probes["novel_relevant"][0]
    payload = {"headline": p.headline, "body": p.body, "stance": p.stance.value}
    scored = _post(base_url + "/api/score", payload)
    assert scored["result"]["score"] >= 0.6 and scored["corpus_size"] == 50
    assert all("headline" in n for n in scored["result"]["nearest"])
    committed = _post(base_url + "/api/score", {**payload, "commit": True})
    assert committed["id"] == "u01" and committed["corpus_size"] == 51
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
    assert [r["id"] for r in saved] == ["u01"] and saved[0]["headline"] == p.headline and saved[0]["admitted"]

    restarted = App(lambda: build_scorer(embedder), user_file)  # simulates a server restart
    assert len(restarted.scorer.corpus) == 51
    assert restarted.scorer.score(p).near_duplicate_of == "u01"


def test_rejected_submissions_are_logged_but_not_replayed(base_url, probes, embedder, user_file):
    p = probes["off_topic"][0]
    resp = _post(base_url + "/api/score", {"headline": p.headline, "body": p.body, "stance": p.stance.value, "commit": True})
    assert resp["result"]["admitted"] is False and resp["id"] is None and resp["corpus_size"] == 50
    assert json.loads(user_file.read_text(encoding="utf-8"))[0]["admitted"] is False
    restarted = App(lambda: build_scorer(embedder), user_file)
    assert len(restarted.scorer.corpus) == 50


def test_corrupt_user_file_is_quarantined_not_fatal(embedder, user_file):
    user_file.write_text("{not json", encoding="utf-8")
    app = App(lambda: build_scorer(embedder), user_file)
    assert app.records == [] and len(app.scorer.corpus) == 50
    assert user_file.with_name("user_submissions.corrupt.json").exists()


def test_bad_saved_records_are_skipped(embedder, user_file):
    user_file.write_text(json.dumps([
        {"id": "u01", "admitted": True, "headline": "", "body": "x", "stance": "support"},  # invalid
        {"id": "u02", "admitted": True, "headline": "Solar canopies on the shuttle lot",
         "body": "The outer shuttle lot is bare asphalt; solar canopies would shade cars.", "stance": "support"},
    ]), encoding="utf-8")
    app = App(lambda: build_scorer(embedder), user_file)
    assert [s.id for s in app.scorer.corpus][-1] == "u02" and len(app.scorer.corpus) == 51


def _raw(base_url, body: bytes, headers=None):
    req = urllib.request.Request(base_url + "/api/score", body, headers or {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req) as resp:
            return resp.status, json.load(resp)
    except urllib.error.HTTPError as e:
        return e.code, json.load(e)


@pytest.mark.parametrize(
    "body",
    [b"not json", b"[1, 2, 3]", b'"a string"', json.dumps({"headline": 5, "body": [], "stance": "support"}).encode(),
     json.dumps({"headline": "ok", "body": "x" * 30, "stance": 3}).encode()],
)
def test_malformed_requests_get_a_400_json_error(base_url, body):
    status, payload = _raw(base_url, body)
    assert status == 400 and "error" in payload


def test_oversized_request_is_refused(base_url):
    status, payload = _raw(base_url, b"{" + b" " * 20_000 + b"}")
    assert status == 400 and "bytes" in payload["error"]


def test_commit_must_be_literally_true(base_url, probes):
    p = probes["novel_relevant"][0]
    resp = _post(base_url + "/api/score", {"headline": p.headline, "body": p.body, "stance": p.stance.value, "commit": "false"})
    assert resp["corpus_size"] == 50  # the string "false" is truthy; it must not commit


def test_reset_forgets_user_submissions(base_url, probes, user_file):
    p = probes["novel_relevant"][2]
    _post(base_url + "/api/score", {"headline": p.headline, "body": p.body, "stance": p.stance.value, "commit": True})
    state = _post(base_url + "/api/reset", {})
    assert state["user_count"] == 0 and state["corpus_size"] == 50 and not user_file.exists()
