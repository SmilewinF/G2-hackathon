"""Error handling and logging: every failure is a specific, explained error (or a logged,
harmless degradation), and every scored input leaves an input line and a result line."""

import importlib.util
import re
import json
import logging
import sqlite3
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pytest

import novelty.data as data
from helpers import HashEmbedder
from novelty import embeddings
from novelty.__main__ import main as cli_main
from novelty.embeddings import CachedEmbedder, GeminiEmbedder, default_embedder, is_transient_error
from novelty.errors import (CalibrationError, DataError, EmbeddingError, NoveltyError, ScoringError,
                            ValidationError)
from novelty.logging_setup import configure_logging, preview, request_id
from novelty.models import Submission
from novelty.scorer import NoveltyScorer, default_signals
from novelty.server import App, _loopback_servers, make_handler
from novelty.signals import Kind, Signal, SignalResult

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(autouse=True)
def restore_novelty_logger():
    """configure_logging() binds handlers to the current stderr and stops propagation; undo it so
    later tests (and caplog) see the logger as it was."""
    logger = logging.getLogger("novelty")
    saved = (logger.level, logger.propagate, list(logger.handlers))
    yield
    for h in logger.handlers:
        if h not in saved[2]:
            h.close()
    logger.setLevel(saved[0])
    logger.propagate = saved[1]
    logger.handlers[:] = saved[2]

# ---------------------------------------------------------------- error hierarchy


def test_errors_keep_their_builtin_bases():
    assert issubclass(ValidationError, ValueError) and issubclass(DataError, ValueError)
    assert issubclass(CalibrationError, ValueError)
    assert issubclass(EmbeddingError, RuntimeError) and issubclass(ScoringError, RuntimeError)
    assert all(issubclass(e, NoveltyError) for e in (ValidationError, DataError, CalibrationError,
                                                     EmbeddingError, ScoringError))


@pytest.mark.parametrize("record, message", [
    ({"headline": "Hi", "stance": "support"}, "missing field(s): body"),
    ("not a dict", "must be an object"),
    ({"headline": "Hi", "body": "long enough body text here", "stance": "support", "id": 7}, "non-empty string"),
])
def test_submission_from_dict_explains_what_is_wrong(record, message):
    with pytest.raises(ValidationError, match=re.escape(message)):
        Submission.from_dict(record)


# ---------------------------------------------------------------- data files


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    for name in ("fixed_content.json", "corpus.json", "off_topic_anchors.json", "probes.json"):
        (tmp_path / name).write_text((ROOT / "data" / name).read_text(encoding="utf-8"), encoding="utf-8")
    monkeypatch.setattr(data, "DATA_DIR", tmp_path)
    return tmp_path


def test_missing_data_file_names_the_file(data_dir):
    (data_dir / "corpus.json").unlink()
    with pytest.raises(DataError, match="not found"):
        data.load_corpus()


def test_invalid_json_names_the_file(data_dir):
    (data_dir / "off_topic_anchors.json").write_text("[1, 2", encoding="utf-8")
    with pytest.raises(DataError, match="off_topic_anchors.json is not valid JSON"):
        data.load_off_topic_anchors()


def test_invalid_corpus_item_names_the_item(data_dir):
    items = json.loads((data_dir / "corpus.json").read_text(encoding="utf-8"))
    items[3]["body"] = "too short"
    (data_dir / "corpus.json").write_text(json.dumps(items), encoding="utf-8")
    with pytest.raises(DataError, match=r"corpus.json item 3 \('c04'\): body must be"):
        data.load_corpus()


def test_duplicate_corpus_ids_are_rejected(data_dir):
    items = json.loads((data_dir / "corpus.json").read_text(encoding="utf-8"))
    items[1]["id"] = items[0]["id"]
    (data_dir / "corpus.json").write_text(json.dumps(items), encoding="utf-8")
    with pytest.raises(DataError, match="duplicate id"):
        data.load_corpus()


def test_wrong_top_level_shape_is_rejected(data_dir):
    (data_dir / "fixed_content.json").write_text("[]", encoding="utf-8")
    with pytest.raises(DataError, match="must contain a JSON dict"):
        data.load_fixed_content()


def test_user_submissions_skip_bad_records_with_a_warning(tmp_path, caplog):
    path = tmp_path / "subs.json"
    path.write_text(json.dumps([
        {"id": "u01", "admitted": True, "headline": "", "body": "x", "stance": "support"},
        {"id": "u02", "admitted": False, "headline": "Fine", "body": "A perfectly fine body text.", "stance": "support"},
        {"id": "u03", "admitted": True, "headline": "Fine", "body": "A perfectly fine body text.", "stance": "support"},
    ]), encoding="utf-8")
    with caplog.at_level(logging.WARNING, logger="novelty.data"):
        subs = data.load_user_submissions(path)
    assert [s.id for s in subs] == ["u03"]
    assert "skipping subs.json item 0" in caplog.text


# ---------------------------------------------------------------- embeddings


class _Broken:
    name = "broken"

    def __init__(self, result=None, exc=None):
        self.result, self.exc = result, exc

    def embed(self, texts):
        if self.exc:
            raise self.exc
        return self.result(len(texts))


@pytest.mark.parametrize("inner", [
    _Broken(result=lambda n: np.full((n, 4), np.nan)),
    _Broken(result=lambda n: np.ones((n + 1, 4))),
    _Broken(result=lambda n: [["x"] * 4] * n),
])
def test_malformed_vectors_raise_embedding_error(tmp_path, inner):
    with pytest.raises(EmbeddingError):
        CachedEmbedder(inner, tmp_path).embed(["some text"])


def test_cache_write_failure_does_not_fail_the_request(tmp_path, caplog):
    emb = CachedEmbedder(HashEmbedder(), tmp_path)

    class ReadOnlyDb:
        def execute(self, sql, *args):
            if sql.startswith(("BEGIN", "INSERT")):
                raise sqlite3.OperationalError("disk I/O error")
            return iter(())

        def executemany(self, *args):
            raise sqlite3.OperationalError("disk I/O error")

    emb._db = ReadOnlyDb()
    with caplog.at_level(logging.WARNING, logger="novelty.embeddings"):
        vecs = emb.embed(["the garage park", "another text"])
    assert vecs.shape == (2, 64)
    assert "cache write failed" in caplog.text


def test_unusable_cache_directory_degrades_to_memory(tmp_path, caplog):
    blocker = tmp_path / "not_a_dir"
    blocker.write_text("x", encoding="utf-8")
    with caplog.at_level(logging.WARNING, logger="novelty.embeddings"):
        emb = CachedEmbedder(HashEmbedder(), blocker)
        assert emb.embed(["works anyway"]).shape == (1, 64)
    assert emb._db is None


def test_cache_counts_hits_and_misses(tmp_path):
    emb = CachedEmbedder(HashEmbedder(), tmp_path)
    emb.embed(["a b", "c d"])
    emb.embed(["a b", "e f"])
    assert (emb.misses, emb.hits) == (3, 1)


def test_unknown_embedder_choice_is_rejected(monkeypatch):
    monkeypatch.setenv("NOVELTY_EMBEDDER", "bogus")
    with pytest.raises(EmbeddingError, match="must be 'local' or 'gemini'"):
        default_embedder()


class _ApiError(Exception):
    def __init__(self, code):
        super().__init__(f"HTTP {code}")
        self.code = code


def test_transient_error_classification():
    assert is_transient_error(_ApiError(429)) and is_transient_error(_ApiError(503))
    assert is_transient_error(TimeoutError()) and not is_transient_error(_ApiError(400))


def _gemini_with(responses, monkeypatch):
    monkeypatch.setattr(embeddings.time, "sleep", lambda s: None)
    g = GeminiEmbedder.__new__(GeminiEmbedder)
    g.name, g._model, g._dimensions = "gemini:test", "m", 4
    calls = iter(responses)

    class Models:
        def embed_content(self, **kwargs):
            r = next(calls)
            if isinstance(r, Exception):
                raise r
            return r

    g._client = type("Client", (), {"models": Models()})()
    return g


def test_gemini_retries_transient_errors_then_succeeds(monkeypatch):
    ok = type("Resp", (), {"embeddings": [type("E", (), {"values": [1.0, 0.0, 0.0, 0.0]})()]})()
    g = _gemini_with([_ApiError(503), _ApiError(429), ok], monkeypatch)
    assert g._call(["x"], None) is ok


def test_gemini_does_not_retry_permanent_errors(monkeypatch):
    g = _gemini_with([_ApiError(400), AssertionError("must not be called again")], monkeypatch)
    with pytest.raises(EmbeddingError, match="HTTP 400"):
        g._call(["x"], None)


# ---------------------------------------------------------------- scorer


ANCHORS = ["the football match was great and the striker scored twice in the final minutes"]


def _toy_corpus():
    return [Submission(headline=f"Garage idea {i}", stance="support" if i % 4 else "oppose",
                       body=f"The council plan for the garage and the park is idea number {i} for the downtown area.")
            for i in range(12)]


class _Exploding(Signal):
    name, kind = "exploding", Kind.MODIFIER

    def evaluate(self, analysis, index):
        raise KeyError("boom")


class _FlakyUpdate(Signal):
    name, kind = "flaky", Kind.MODIFIER

    def __init__(self):
        self.fits = 0

    def fit(self, index):
        self.fits += 1

    def update(self, index, added):
        if self.fits:
            raise RuntimeError("incremental path broken")
        self.fit(index)

    def evaluate(self, analysis, index):
        return SignalResult(1.0)


def _toy_scorer(extra):
    from novelty.data import load_fixed_content
    from novelty.scorer import ScorerConfig

    return NoveltyScorer(load_fixed_content(), _toy_corpus(), HashEmbedder(), ANCHORS,
                         signals=[*default_signals(ScorerConfig()), extra])


def test_failing_signal_raises_scoring_error_naming_it():
    scorer = _toy_scorer(_Exploding())
    with pytest.raises(ScoringError, match="signal 'exploding' failed"):
        scorer.score(Submission(headline="x", body="the garage park plan downtown is good", stance="mixed"))


def test_failed_incremental_update_falls_back_to_a_full_fit(caplog):
    flaky = _FlakyUpdate()
    scorer = _toy_scorer(flaky)
    with caplog.at_level(logging.ERROR, logger="novelty.scorer"):
        scorer.add(Submission(headline="More", body="The garage could become a covered market downtown.",
                              stance="mixed", id="m1"))
    assert flaky.fits == 2 and "recalibrating from scratch" in caplog.text


def test_every_scored_input_logs_an_input_and_a_result_line(scorer, caplog):
    sub = Submission(headline="Solar canopies", body="Solar canopies over the outer shuttle lot would shade cars.",
                     stance="support")
    with caplog.at_level(logging.INFO, logger="novelty.scorer"):
        scorer.score(sub)
        scorer.submit(sub)
    lines = [r.getMessage() for r in caplog.records]
    assert any(line.startswith('score input: id=- stance=support words=10 headline="Solar canopies"') for line in lines)
    result = next(line for line in lines if line.startswith("score result:"))
    assert "score=" in result and "novelty" in result and "gate" in result and " ms" in result
    assert any(line.startswith("submit result:") and "added to corpus" in line for line in lines)


def test_debug_logs_the_full_calculation(scorer, caplog):
    with caplog.at_level(logging.DEBUG, logger="novelty.scorer"):
        scorer.score(Submission(headline="Parkng", body="Withot the garaje peple cant park near the shops.", stance="oppose"))
    calc = next(r.getMessage() for r in caplog.records if r.getMessage().startswith("calculation:"))
    assert "whole_text=" in calc and "relevance=" in calc and "garaje->garage" in calc


# ---------------------------------------------------------------- logging setup


def test_configure_logging_is_idempotent_and_tolerates_bad_levels():
    logger = configure_logging("NOT_A_LEVEL")
    assert logger.level == logging.INFO
    configure_logging("DEBUG")
    owned = [h for h in logger.handlers if getattr(h, "_novelty_handler", False)]
    assert len(owned) == 1 and logger.level == logging.DEBUG
    configure_logging("WARNING")


def test_log_lines_carry_the_request_id(capsys):
    logger = configure_logging("INFO")
    token = request_id.set("r000042")
    try:
        logging.getLogger("novelty.test").info("hello")
    finally:
        request_id.reset(token)
    assert "[r000042] novelty.test: hello" in capsys.readouterr().err
    configure_logging("WARNING")
    assert logger is logging.getLogger("novelty")


def test_preview_flattens_and_truncates():
    assert preview("a\n  b") == "a b"
    assert preview("x" * 100, 20) == "x" * 17 + "..."


# ---------------------------------------------------------------- server


@pytest.fixture
def app(base_scorer, tmp_path):
    return App(base_scorer.fork, tmp_path / "user_submissions.json")


@pytest.fixture
def served(app):
    from http.server import ThreadingHTTPServer
    import threading

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(app))
    threading.Thread(target=httpd.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True).start()
    yield app, f"http://127.0.0.1:{httpd.server_port}"
    httpd.shutdown()


def _request(url, payload=None):
    body = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(url, body, {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req) as resp:
            return resp.status, json.load(resp)
    except urllib.error.HTTPError as e:
        return e.code, json.load(e)


GOOD = {"headline": "Solar canopies", "body": "Solar canopies over the outer shuttle lot would shade cars.", "stance": "support"}


def test_embedding_outage_is_a_503_with_request_id(served, monkeypatch):
    app, url = served

    def down(sub):
        raise EmbeddingError("backend unreachable")

    monkeypatch.setattr(app.scorer, "score", down)
    status, payload = _request(url + "/api/score", GOOD)
    assert status == 503 and "backend unreachable" in payload["error"] and payload["request_id"].startswith("r")


def test_unexpected_error_is_a_500_that_points_to_the_log(served, monkeypatch):
    app, url = served
    monkeypatch.setattr(app.scorer, "score", lambda sub: {}["missing"])
    status, payload = _request(url + "/api/score", GOOD)
    assert status == 500 and "see server log" in payload["error"] and payload["request_id"] in payload["error"]


def test_failed_save_still_answers_and_says_so(served):
    app, url = served
    app.user_file = app.user_file.parent / "no_such_dir" / "subs.json"
    status, payload = _request(url + "/api/score", {**GOOD, "commit": True})
    assert status == 200 and payload["saved"] is False and len(app.records) == 1


def test_unknown_route_and_query_strings(served):
    _, url = served
    status, payload = _request(url + "/nope")
    assert status == 404 and payload["request_id"]
    assert _request(url + "/api/context?x=1")[0] == 200


def test_port_in_use_is_a_clear_error(served):
    _, url = served
    port = int(url.rsplit(":", 1)[1])
    with pytest.raises(NoveltyError, match=f"cannot listen on 127.0.0.1:{port}"):
        _loopback_servers("127.0.0.1", port, make_handler(None))


# ---------------------------------------------------------------- CLI and scripts


def test_cli_reports_validation_errors_with_exit_code_2(capsys):
    assert cli_main(["score", "--headline", "", "--body", "short", "--stance", "support"]) == 2
    assert "error: headline is required" in capsys.readouterr().err


def test_cli_success_returns_0(capsys):
    assert cli_main(["--log-level", "WARNING", "score", "--headline", "Solar", "--body",
                     "Solar canopies over the outer shuttle lot would shade cars.", "--stance", "support"]) == 0
    assert capsys.readouterr().out.startswith("score ")


@pytest.fixture
def generate_corpus():
    spec = importlib.util.spec_from_file_location("generate_corpus", ROOT / "scripts" / "generate_corpus.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_generator_refuses_to_overwrite_without_force(generate_corpus, tmp_path):
    out = tmp_path / "corpus.json"
    out.write_text("[]", encoding="utf-8")
    assert generate_corpus.main(["--out", str(out)]) == 2 and out.read_text(encoding="utf-8") == "[]"


def test_generator_validates_the_model_response(generate_corpus):
    with pytest.raises(generate_corpus.GenerationError, match="not valid JSON"):
        generate_corpus._rows("not json")
    rows = generate_corpus._rows(json.dumps([
        {"headline": "Ok", "body": "A valid body that is long enough.", "stance": "support"},
        {"headline": "", "body": "invalid", "stance": "support"},
    ]))
    assert [r["headline"] for r in rows] == ["Ok"]


def test_generator_retries_transient_api_errors(generate_corpus, monkeypatch):
    monkeypatch.setattr(generate_corpus.time, "sleep", lambda s: None)
    replies = iter([_ApiError(503), type("R", (), {"text": "[]"})()])

    class Models:
        def generate_content(self, **kwargs):
            r = next(replies)
            if isinstance(r, Exception):
                raise r
            return r

    class Types:
        GenerateContentConfig = staticmethod(lambda **kw: kw)

    client = type("Client", (), {"models": Models()})()
    assert generate_corpus._generate(client, Types, "m", "prompt") == "[]"
