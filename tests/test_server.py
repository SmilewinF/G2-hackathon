"""The web UI's JSON API, exercised over a real socket."""

import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from novelty.server import make_handler


@pytest.fixture
def base_url(scorer):
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(scorer))
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
