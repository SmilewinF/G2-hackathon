"""Minimal local web UI for trying the scorer. Standard library only.

    python -m novelty serve [--port 8000]

GET  /              the page (novelty/static/index.html)
GET  /api/context   fixed content, corpus size and example probes
POST /api/score     {"headline", "body", "stance", "commit": bool} -> ScoreBreakdown JSON
                    commit=true also adds the submission to the in-memory corpus (submit()).
"""

from __future__ import annotations

import dataclasses
import json
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .data import build_scorer, load_probes
from .models import Stance, Submission
from .scorer import NoveltyScorer

STATIC = Path(__file__).resolve().parent / "static"
MAX_BODY_BYTES = 16_384


def make_handler(scorer: NoveltyScorer) -> type[BaseHTTPRequestHandler]:
    lock = threading.Lock()  # submit() mutates the corpus
    counter = {"n": 0}

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

        def do_GET(self) -> None:
            if self.path == "/":
                self._send(HTTPStatus.OK, (STATIC / "index.html").read_bytes(), "text/html; charset=utf-8")
            elif self.path == "/api/context":
                with lock:
                    corpus = {s.id: s.headline for s in scorer.corpus}
                self._json(HTTPStatus.OK, {
                    "fixed": dataclasses.asdict(scorer.fixed),
                    "corpus": corpus,
                    "stances": [s.value for s in Stance],
                    "probes": load_probes(),
                    "embedder": scorer.embedder.name,
                })
            else:
                self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})

        def do_POST(self) -> None:
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
            with lock:
                if data.get("commit"):
                    counter["n"] += 1
                    sub = dataclasses.replace(sub, id=f"u{counter['n']:02d}")
                    result = scorer.submit(sub)
                else:
                    result = scorer.score(sub)
                corpus = {s.id: s.headline for s in scorer.corpus}
            self._json(HTTPStatus.OK, {"result": dataclasses.asdict(result), "id": sub.id, "corpus": corpus})

    return Handler


def serve(port: int = 8000, host: str = "127.0.0.1") -> None:
    print("loading model and corpus...")
    scorer = build_scorer()
    httpd = ThreadingHTTPServer((host, port), make_handler(scorer))
    print(f"open http://{host}:{port}  (embedder: {scorer.embedder.name}, Ctrl+C to stop)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
