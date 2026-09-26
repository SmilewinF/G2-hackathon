"""Minimal local web UI for trying the scorer. Standard library only.

    python -m novelty serve [--port 8000]

GET  /              the page (novelty/static/index.html)
GET  /api/context   fixed content, corpus headlines, example probes
POST /api/score     {"headline", "body", "stance", "commit": bool} -> ScoreBreakdown JSON
                    commit=true also adds the submission to the corpus (submit()) and saves it
                    to data/user_submissions.json, which is replayed on the next start.
POST /api/reset     forget all user submissions (deletes that file, rebuilds the scorer)

User submissions are kept out of data/corpus.json on purpose: the tests and README numbers are
calibrated against that fixed 50-item seed corpus.
"""

from __future__ import annotations

import dataclasses
import json
import threading
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable

from .data import DATA_DIR, build_scorer, load_probes
from .models import Stance, Submission
from .scorer import NoveltyScorer

STATIC = Path(__file__).resolve().parent / "static"
USER_FILE = DATA_DIR / "user_submissions.json"
MAX_BODY_BYTES = 16_384


class App:
    """Scorer plus the persisted log of user submissions. All access goes through ``lock``."""

    def __init__(self, scorer_factory: Callable[[], NoveltyScorer], user_file: Path) -> None:
        self.lock = threading.Lock()
        self._factory = scorer_factory
        self.user_file = user_file
        self._load()

    def _load(self) -> None:
        self.scorer = self._factory()
        self.records: list[dict] = []
        if self.user_file.exists():
            self.records = json.loads(self.user_file.read_text(encoding="utf-8"))
        for rec in self.records:
            self.scorer.add(Submission.from_dict(rec), on_topic=rec["on_topic"])

    def _save(self) -> None:
        self.user_file.write_text(json.dumps(self.records, indent=2) + "\n", encoding="utf-8")

    def corpus_headlines(self) -> dict[str, str]:
        return {s.id: s.headline for s in self.scorer.corpus}

    def submit(self, sub: Submission):
        sub = dataclasses.replace(sub, id=f"u{len(self.records) + 1:02d}")
        result = self.scorer.submit(sub)
        self.records.append({
            "id": sub.id,
            "stance": sub.stance.value,
            "headline": sub.headline,
            "body": sub.body,
            "score": result.score,
            "on_topic": result.relevance_gate > 0.0,
            "submitted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        })
        self._save()
        return sub, result

    def reset(self) -> None:
        self.user_file.unlink(missing_ok=True)
        self._load()


def make_handler(app: App) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):  # keep the console quiet
            pass

        def _send(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _json(self, status: int, payload) -> None:
            self._send(status, json.dumps(payload).encode(), "application/json")

        def _state(self) -> dict:
            return {"corpus": app.corpus_headlines(), "user_count": len(app.records)}

        def do_GET(self) -> None:
            if self.path == "/":
                self._send(HTTPStatus.OK, (STATIC / "index.html").read_bytes(), "text/html; charset=utf-8")
            elif self.path == "/api/context":
                with app.lock:
                    self._json(HTTPStatus.OK, {
                        **self._state(),
                        "fixed": dataclasses.asdict(app.scorer.fixed),
                        "stances": [s.value for s in Stance],
                        "probes": load_probes(),
                        "embedder": app.scorer.embedder.name,
                    })
            else:
                self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})

        def do_POST(self) -> None:
            if self.path == "/api/reset":
                with app.lock:
                    app.reset()
                    return self._json(HTTPStatus.OK, self._state())
            if self.path != "/api/score":
                return self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
            length = int(self.headers.get("Content-Length", 0))
            if length > MAX_BODY_BYTES:
                return self._json(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"error": "request too large"})
            try:
                data = json.loads(self.rfile.read(length) or b"{}")
                sub = Submission(headline=data.get("headline", ""), body=data.get("body", ""),
                                 stance=data.get("stance", ""))
            except (ValueError, TypeError) as e:
                return self._json(HTTPStatus.BAD_REQUEST, {"error": str(e)})
            with app.lock:
                if data.get("commit"):
                    sub, result = app.submit(sub)
                else:
                    result = app.scorer.score(sub)
                self._json(HTTPStatus.OK, {**self._state(), "result": dataclasses.asdict(result), "id": sub.id})

    return Handler


def serve(port: int = 8000, host: str = "127.0.0.1", user_file: Path = USER_FILE) -> None:
    print("loading model and corpus...")
    app = App(build_scorer, user_file)
    httpd = ThreadingHTTPServer((host, port), make_handler(app))
    print(f"loaded {len(app.records)} saved user submission(s) from {user_file.name}")
    print(f"open http://{host}:{port}  (embedder: {app.scorer.embedder.name}, Ctrl+C to stop)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
