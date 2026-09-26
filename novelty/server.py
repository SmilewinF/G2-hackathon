"""Minimal local web UI for trying the scorer. Standard library only.

    python -m novelty serve [--port 8000]

GET  /              the page (novelty/static/index.html)
GET  /api/context   fixed content, corpus headlines, example probes
POST /api/score     {"headline", "body", "stance", "commit": bool} -> ScoreBreakdown JSON
                    commit=true runs submit(): the attempt is logged to data/user_submissions.json
                    and, if the admission policy accepts it, added to the corpus. Admitted entries
                    are replayed on the next start.
POST /api/reset     forget all user submissions (deletes that file, rebuilds the scorer)

User submissions are kept out of data/corpus.json on purpose: the tests and README numbers are
calibrated against that fixed 50-item seed corpus.
"""

from __future__ import annotations

import dataclasses
import json
import os
import sys
import threading
import traceback
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


class BadRequest(ValueError):
    pass


class App:
    """Scorer plus the persisted log of user submissions. All access goes through ``lock``."""

    def __init__(self, scorer_factory: Callable[[], NoveltyScorer], user_file: Path) -> None:
        self.lock = threading.Lock()
        self._factory = scorer_factory
        self.user_file = user_file
        self._load()

    # ------------------------------------------------------------------ persistence

    def _load(self) -> None:
        self.scorer = self._factory()
        self.records = self._read_records()
        for rec in self.records:
            if not rec.get("admitted"):
                continue
            try:
                self.scorer.add(Submission.from_dict(rec))
            except (KeyError, TypeError, ValueError) as e:
                print(f"skipping saved submission {rec.get('id')!r}: {e}", file=sys.stderr)

    def _read_records(self) -> list[dict]:
        if not self.user_file.exists():
            return []
        try:
            records = json.loads(self.user_file.read_text(encoding="utf-8"))
            if not isinstance(records, list) or not all(isinstance(r, dict) for r in records):
                raise ValueError("expected a JSON list of objects")
            return records
        except (ValueError, UnicodeDecodeError) as e:  # JSONDecodeError is a ValueError
            backup = self.user_file.with_name(self.user_file.stem + ".corrupt.json")
            os.replace(self.user_file, backup)
            print(f"{self.user_file.name} is unreadable ({e}); moved it to {backup.name}", file=sys.stderr)
            return []

    def _save(self) -> None:
        tmp = self.user_file.with_name(self.user_file.name + ".tmp")
        tmp.write_text(json.dumps(self.records, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, self.user_file)  # atomic: a crash mid-write cannot corrupt the log

    # ------------------------------------------------------------------ operations

    @property
    def admitted_count(self) -> int:
        return sum(bool(r.get("admitted")) for r in self.records)

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
            "admitted": result.admitted,
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

        # -------------------------------------------------------------- plumbing

        def _send(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _json(self, status: int, payload) -> None:
            self._send(status, json.dumps(payload).encode(), "application/json")

        def _read_json(self) -> dict:
            try:
                length = int(self.headers.get("Content-Length", 0))
            except ValueError:
                raise BadRequest("invalid Content-Length") from None
            if not 0 <= length <= MAX_BODY_BYTES:
                raise BadRequest(f"request body must be 0-{MAX_BODY_BYTES} bytes")
            try:
                data = json.loads(self.rfile.read(length) or b"{}")
            except (json.JSONDecodeError, UnicodeDecodeError):
                raise BadRequest("request body is not valid JSON") from None
            if not isinstance(data, dict):
                raise BadRequest("request body must be a JSON object")
            return data

        def _guarded(self, route: Callable[[], None]) -> None:
            """Every failure becomes a JSON error response instead of a dropped connection."""
            try:
                route()
            except BadRequest as e:
                self._json(HTTPStatus.BAD_REQUEST, {"error": str(e)})
            except (OSError, RuntimeError) as e:  # embedder / network / disk failure
                traceback.print_exc()
                self._json(HTTPStatus.SERVICE_UNAVAILABLE, {"error": f"scoring unavailable: {e}"})
            except Exception:
                traceback.print_exc()
                self._json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": "internal error, see server log"})

        def _state(self) -> dict:
            return {"corpus": app.corpus_headlines(), "user_count": app.admitted_count}

        # -------------------------------------------------------------- routes

        def do_GET(self) -> None:
            self._guarded(self._get)

        def do_POST(self) -> None:
            self._guarded(self._post)

        def _get(self) -> None:
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

        def _post(self) -> None:
            if self.path == "/api/reset":
                with app.lock:
                    app.reset()
                    self._json(HTTPStatus.OK, self._state())
                return
            if self.path != "/api/score":
                self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
                return
            data = self._read_json()
            try:
                sub = Submission(headline=data.get("headline", ""), body=data.get("body", ""),
                                 stance=data.get("stance", ""))
            except ValueError as e:
                raise BadRequest(str(e)) from None
            with app.lock:
                if data.get("commit") is True:
                    sub, result = app.submit(sub)
                else:
                    result = app.scorer.score(sub)
                self._json(HTTPStatus.OK, {
                    **self._state(),
                    "result": dataclasses.asdict(result),
                    "id": sub.id if result.admitted else None,
                })

    return Handler


def serve(port: int = 8000, host: str = "127.0.0.1", user_file: Path = USER_FILE) -> None:
    print("loading model and corpus...")
    app = App(build_scorer, user_file)
    httpd = ThreadingHTTPServer((host, port), make_handler(app))
    print(f"loaded {app.admitted_count} saved user submission(s) from {user_file.name}")
    print(f"open http://{host}:{port}  (embedder: {app.scorer.embedder.name}, Ctrl+C to stop)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
