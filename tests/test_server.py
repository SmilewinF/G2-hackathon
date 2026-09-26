"""The web UI's JSON API, exercised over a real socket.

The server under test starts from the same corpus as the rest of the suite (seed + admitted
web-UI submissions in full mode) and writes to its own temporary submissions file.
"""

import json
import urllib.error
import urllib.request

import pytest

from helpers import already_submitted, request_json
from novelty.server import App


def _post(url, payload):
    req = urllib.request.Request(url, json.dumps(payload).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as resp:
        return json.load(resp)


def _fresh_probe(base_scorer, probes, idx):
    p = probes["novel_relevant"][idx]
    if already_submitted(base_scorer, p):
        pytest.skip(f"probe {p.id} was already submitted through the web UI")
    return p


def _payload(p, **extra):
    return {"headline": p.headline, "body": p.body, "stance": p.stance.value, **extra}


def test_page_and_context_are_served(base_url, corpus_size):
    with urllib.request.urlopen(base_url + "/") as resp:
        assert b"Novelty Scorer" in resp.read()
    with urllib.request.urlopen(base_url + "/api/context") as resp:
        ctx = json.load(resp)
    assert ctx["corpus_size"] == corpus_size and ctx["stances"] and ctx["probes"]["novel_relevant"]


def test_score_does_not_change_corpus_but_commit_does(base_url, base_scorer, probes, corpus_size):
    p = _fresh_probe(base_scorer, probes, 0)
    scored = _post(base_url + "/api/score", _payload(p))
    assert scored["result"]["score"] >= 0.6 and scored["corpus_size"] == corpus_size
    assert all("headline" in n for n in scored["result"]["nearest"])
    committed = _post(base_url + "/api/score", _payload(p, commit=True))
    uid = committed["id"]
    assert uid and uid.startswith("u") and committed["corpus_size"] == corpus_size + 1
    again = _post(base_url + "/api/score", _payload(p))
    assert again["result"]["near_duplicate_of"] == uid and again["result"]["score"] == 0.0


def test_invalid_submission_returns_400(base_url):
    with pytest.raises(urllib.error.HTTPError) as err:
        _post(base_url + "/api/score", {"headline": "", "body": "short", "stance": "maybe"})
    assert err.value.code == 400


def test_committed_submissions_are_saved_and_reloaded(base_url, base_scorer, probes, user_file, corpus_size):
    p = _fresh_probe(base_scorer, probes, 1)
    _post(base_url + "/api/score", _payload(p, commit=True))
    saved = json.loads(user_file.read_text(encoding="utf-8"))
    assert len(saved) == 1 and saved[0]["headline"] == p.headline and saved[0]["admitted"]

    restarted = App(base_scorer.fork, user_file)  # simulates a server restart
    assert len(restarted.scorer.corpus) == corpus_size + 1
    assert restarted.scorer.score(p).near_duplicate_of == saved[0]["id"]


def test_rejected_submissions_are_logged_but_not_replayed(base_url, base_scorer, probes, user_file, corpus_size):
    p = probes["off_topic"][0]
    resp = _post(base_url + "/api/score", _payload(p, commit=True))
    assert resp["result"]["admitted"] is False and resp["id"] is None and resp["corpus_size"] == corpus_size
    assert json.loads(user_file.read_text(encoding="utf-8"))[0]["admitted"] is False
    restarted = App(base_scorer.fork, user_file)
    assert len(restarted.scorer.corpus) == corpus_size


def test_corrupt_user_file_is_quarantined_not_fatal(base_scorer, user_file, corpus_size):
    user_file.write_text("{not json", encoding="utf-8")
    app = App(base_scorer.fork, user_file)
    assert app.records == [] and len(app.scorer.corpus) == corpus_size
    assert user_file.with_name("user_submissions.corrupt.json").exists()


def test_bad_saved_records_are_skipped(base_scorer, user_file, corpus_size):
    user_file.write_text(json.dumps([
        {"id": "t01", "admitted": True, "headline": "", "body": "x", "stance": "support"},  # invalid
        {"id": "t02", "admitted": True, "headline": "Solar canopies on the shuttle lot",
         "body": "The outer shuttle lot is bare asphalt; solar canopies would shade cars.", "stance": "support"},
    ]), encoding="utf-8")
    app = App(base_scorer.fork, user_file)
    assert [s.id for s in app.scorer.corpus][-1] == "t02" and len(app.scorer.corpus) == corpus_size + 1


def _raw(base_url, body: bytes):
    return request_json(base_url + "/api/score", body)


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


def test_commit_must_be_literally_true(base_url, probes, corpus_size):
    p = probes["novel_relevant"][0]
    resp = _post(base_url + "/api/score", _payload(p, commit="false"))
    assert resp["corpus_size"] == corpus_size  # the string "false" is truthy; it must not commit


def test_reset_forgets_user_submissions(base_url, probes, user_file, corpus_size):
    p = probes["novel_relevant"][2]
    _post(base_url + "/api/score", _payload(p, commit=True))
    state = _post(base_url + "/api/reset", {})
    assert state["user_count"] == 0 and state["corpus_size"] == corpus_size and not user_file.exists()
