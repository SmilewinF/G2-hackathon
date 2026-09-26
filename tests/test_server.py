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
    with err.value:
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


# ---------------------------------------------------------------- cross-site requests and hardening


def _send(base_url, method, path, body=None, headers=None):
    """(status, headers, JSON payload) of a raw request with exactly these headers."""
    import http.client
    from urllib.parse import urlsplit

    url = urlsplit(base_url)
    conn = http.client.HTTPConnection(url.hostname, url.port, timeout=30)
    try:
        conn.request(method, path, body=body, headers={"Host": url.netloc, **(headers or {})})
        resp = conn.getresponse()
        raw = resp.read()
        return resp.status, dict(resp.getheaders()), json.loads(raw) if raw else None
    finally:
        conn.close()


def test_cross_site_posts_cannot_commit_or_reset(base_url, probes, user_file):
    """A text/plain POST needs no CORS preflight, so any web page could commit or wipe
    submissions while the server ran; both used to succeed."""
    body = json.dumps(_payload(probes["novel_relevant"][0], commit=True))
    status, _, _ = _send(base_url, "POST", "/api/score", body, {"Content-Type": "text/plain"})
    assert status == 415
    status, _, _ = _send(base_url, "POST", "/api/score", body,
                         {"Content-Type": "application/json", "Origin": "http://evil.example"})
    assert status == 403
    user_file.write_text("[]", encoding="utf-8")
    status, _, _ = _send(base_url, "POST", "/api/reset", None, {"Origin": "http://evil.example"})
    assert status == 403 and user_file.exists()
    status, _, _ = _send(base_url, "POST", "/api/reset")  # no JSON content type
    assert status == 415 and user_file.exists()


def test_foreign_host_header_is_refused(base_url):
    """DNS rebinding: a page on another domain that resolves to 127.0.0.1 must not read the API."""
    status, _, payload = _send(base_url, "GET", "/api/context", headers={"Host": "attacker.example"})
    assert status == 403 and "Host" in payload["error"]


def test_same_origin_page_requests_still_work(base_url, probes):
    origin = base_url.replace("/api", "")
    status, _, payload = _send(base_url, "POST", "/api/score", json.dumps(_payload(probes["novel_relevant"][0])),
                               {"Content-Type": "application/json; charset=utf-8", "Origin": origin})
    assert status == 200 and payload["result"]["score"] > 0


def test_responses_carry_security_headers(base_url):
    _, headers, _ = _send(base_url, "GET", "/api/context")
    assert "frame-ancestors 'none'" in headers["Content-Security-Policy"]
    assert headers["X-Content-Type-Options"] == "nosniff" and headers["X-Frame-Options"] == "DENY"


@pytest.mark.parametrize("method", ["PUT", "DELETE", "PATCH", "OPTIONS"])
def test_other_methods_get_a_json_405(base_url, method):
    status, headers, payload = _send(base_url, method, "/api/score", "{}")
    assert status == 405 and headers["Allow"] == "GET, POST" and "error" in payload


def test_json_number_too_long_to_parse_is_a_400(base_url):
    status, payload = _raw(base_url, b'{"headline": ' + b"9" * 5000 + b"}")  # was: 500 (ValueError)
    assert status == 400 and "error" in payload


def test_saved_records_that_cannot_be_replayed_do_not_stop_the_server(base_scorer, user_file, corpus_size):
    """A record without an id, or with the article's id, used to raise out of App() at startup;
    "admitted": "no" counted as admitted, and user_count included records the replay skipped."""
    fine = {"headline": "Solar canopies on the shuttle lot", "stance": "support",
            "body": "The outer shuttle lot is bare asphalt; solar canopies would shade cars."}
    user_file.write_text(json.dumps([
        {**fine, "admitted": True},  # no id
        {**fine, "id": "article", "admitted": True},
        {**fine, "id": "t01", "admitted": "no"},
        {**fine, "id": "t02", "admitted": True},
    ]), encoding="utf-8")
    app = App(base_scorer.fork, user_file)
    assert [s.id for s in app.scorer.corpus][-1] == "t02" and len(app.scorer.corpus) == corpus_size + 1
    assert app.admitted_count == 1
