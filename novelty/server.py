"""Minimal local web UI for trying the scorer. Standard library only.

    python -m novelty serve [--port 8000]

GET  /              the page (novelty/static/index.html)
GET  /api/context   fixed content, stances, corpus size, user count, UI examples (two per stance),
                    labelled probes, embedder name
POST /api/score     {"headline", "body", "stance", "commit": bool} -> {"corpus_size", "user_count",
                    "result": ScoreBreakdown JSON with a "headline" added to each nearest entry,
                    "id": the new id if admitted else null, "saved": bool (only when committed)}
                    commit=true runs submit(): the attempt is logged to data/user_submissions.json
                    and, if the admission policy accepts it, added to the corpus. Admitted entries
                    are replayed on the next start.
POST /api/reset     forget all user submissions (deletes that file, forks the pristine seed-corpus
                    scorer again instead of rebuilding it)

User submissions are kept out of data/corpus.json on purpose: the tests and README numbers are
calibrated against that fixed 50-item seed corpus.

Only this machine's own page may use the API: every request must carry a loopback Host header
(127.0.0.1, localhost or [::1] with this port), a request with an Origin must come from one of
those, and POST bodies must be Content-Type: application/json. Together these stop other web
sites from posting to the server (a text/plain or form POST needs no CORS preflight) and DNS
rebinding pages from reading it. Responses carry a restrictive Content-Security-Policy and
frame-ancestors 'none'.

Errors always come back as JSON {"error": ..., "request_id": ...}: 400 for bad input, 403 for a
foreign Host or Origin, 404 for an unknown path, 405 for other methods, 415 for a non-JSON POST,
503 when the embedding backend is unavailable or an OSError (disk or network trouble) escapes a
route, 500 otherwise (details in the server log under the same request id).
"""

from __future__ import annotations

import dataclasses
import itertools
import json
import logging
import os
import socket
import threading
import time
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable
from urllib.parse import urlsplit

from .data import USER_FILE, build_scorer, load_examples, load_probes, replayable_submissions, write_json_atomic
from .errors import EmbeddingError, NoveltyError, ValidationError
from .index import ARTICLE_ID
from .logging_setup import configure_logging, printable, request_id
from .models import ScoreBreakdown, Stance, Submission
from .scorer import NoveltyScorer

log = logging.getLogger(__name__)

STATIC = Path(__file__).resolve().parent / "static"
MAX_BODY_BYTES = 16_384
_request_ids = itertools.count(1)


class BadRequest(ValidationError):
    status = HTTPStatus.BAD_REQUEST


class Forbidden(BadRequest):
    """Cross-site request or foreign Host header: DNS rebinding and CSRF protection."""

    status = HTTPStatus.FORBIDDEN


class UnsupportedMediaType(BadRequest):
    status = HTTPStatus.UNSUPPORTED_MEDIA_TYPE


LOOPBACK_NAMES = ("127.0.0.1", "localhost", "[::1]")
# The page is self-contained (inline script and style, same-origin fetches only).
SECURITY_HEADERS = {
    "Content-Security-Policy": "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; "
                               "connect-src 'self'; img-src 'self' data:; base-uri 'none'; form-action 'self'; "
                               "frame-ancestors 'none'",
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
}


class App:
    """Scorer plus the persisted log of user submissions. All access goes through ``lock``."""

    def __init__(self, scorer_factory: Callable[[], NoveltyScorer], user_file: Path) -> None:
        self.lock = threading.Lock()
        self._pristine = scorer_factory()  # seed corpus only; reset() forks it instead of rebuilding
        self.user_file = user_file
        self.probes = load_probes()  # static data for the page: read once, not per request
        self.examples = load_examples()
        self._load()

    # ------------------------------------------------------------------ persistence

    def _load(self) -> None:
        self.scorer = self._pristine.fork()
        self.records = self._read_records()
        replay = replayable_submissions(self.records, self.user_file.name, reserved={s.id for s in self.scorer.corpus})
        self.scorer.add_many(replay)  # one recalibration for the whole log, not one per record
        self._user_ids = {s.id for s in replay}
        if replay:
            log.info("replayed %d saved submission(s) from %s", len(replay), self.user_file.name)

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
            try:
                os.replace(self.user_file, backup)
                log.warning("%s is unreadable (%s); moved it to %s", self.user_file.name, e, backup.name)
            except OSError as move_error:
                log.error("%s is unreadable (%s) and could not be moved aside (%s); starting empty",
                          self.user_file.name, e, move_error)
            return []
        except OSError as e:
            log.error("cannot read %s (%s); starting without saved submissions", self.user_file, e)
            return []

    def _save(self) -> bool:
        try:
            write_json_atomic(self.user_file, self.records)  # a crash mid-write cannot corrupt the log
            return True
        except OSError as e:
            # The submission is already scored and (if admitted) in the corpus; losing the disk
            # copy must not fail the request. The next successful save writes it too.
            log.error("could not save %s: %s", self.user_file, e)
            return False

    # ------------------------------------------------------------------ operations

    @property
    def admitted_count(self) -> int:
        """User submissions in the corpus now (saved records the replay skipped do not count)."""
        return len(self._user_ids)

    def headline(self, entry_id: str) -> str:
        if entry_id == ARTICLE_ID:
            return self.scorer.fixed.title
        entry = self.scorer.index.get(entry_id)
        return entry.submission.headline if entry and entry.submission else ""

    def _next_id(self) -> str:
        n = len(self.records) + 1
        while self.scorer.index.get(f"u{n:02d}") is not None:
            n += 1
        return f"u{n:02d}"

    def submit(self, sub: Submission) -> tuple[Submission, ScoreBreakdown, bool]:
        sub = dataclasses.replace(sub, id=self._next_id())
        result = self.scorer.submit(sub)
        if result.admitted:
            self._user_ids.add(sub.id)
        self.records.append({
            "id": sub.id,
            "stance": sub.stance.value,
            "headline": sub.headline,
            "body": sub.body,
            "score": result.score,
            "admitted": result.admitted,
            "submitted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        })
        return sub, result, self._save()

    def reset(self) -> None:
        try:
            self.user_file.unlink(missing_ok=True)
        except OSError as e:
            raise NoveltyError(f"could not delete {self.user_file.name}: {e}") from e
        self._load()
        log.info("user submissions reset; corpus back to %d", len(self.scorer.corpus))


def make_handler(app: App) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        # Seconds a client may leave the socket idle; a half-sent request no longer holds a thread forever.
        timeout = 15

        def log_message(self, fmt, *args):  # replaced by our own access log below
            pass

        # -------------------------------------------------------------- plumbing

        def _send(self, status: int, body: bytes, content_type: str) -> None:
            self._status = status
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            # Never reuse a stale page or response: an old copy of the page calling a newer API
            # broke with "Cannot convert undefined or null to object".
            self.send_header("Cache-Control", "no-store")
            for name, value in SECURITY_HEADERS.items():
                self.send_header(name, value)
            if status == HTTPStatus.METHOD_NOT_ALLOWED:
                self.send_header("Allow", "GET, POST")
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)

        def _json(self, status: int, payload) -> None:
            self._send(status, json.dumps(payload).encode(), "application/json")

        def _error(self, status: int, message: str) -> None:
            self._json(status, {"error": message, "request_id": request_id.get()})

        def _read_json(self) -> dict:
            content_type = self.headers.get("Content-Type", "").split(";")[0].strip().lower()
            if content_type != "application/json":
                # Browsers send text/plain and form posts cross-site without asking; JSON makes
                # them ask first (a CORS preflight, which this server never grants).
                raise UnsupportedMediaType("POST requests must be Content-Type: application/json")
            try:
                length = int(self.headers.get("Content-Length", 0))
            except ValueError:
                raise BadRequest("invalid Content-Length") from None
            if not 0 <= length <= MAX_BODY_BYTES:
                raise BadRequest(f"request body must be 0-{MAX_BODY_BYTES} bytes")
            try:
                data = json.loads(self.rfile.read(length) or b"{}")
            except (ValueError, UnicodeDecodeError, RecursionError):  # ValueError: bad JSON or a 4,300+ digit number
                raise BadRequest("request body is not valid JSON") from None
            if not isinstance(data, dict):
                raise BadRequest("request body must be a JSON object")
            return data

        def _check_origin(self) -> None:
            """Only this machine's own page may call the API. The Host check stops DNS rebinding
            (a remote page whose domain resolves to 127.0.0.1 reading responses); the Origin check
            stops other sites posting to it."""
            port = self.server.server_address[1]
            hosts = {f"{name}:{port}" for name in LOOPBACK_NAMES}
            if port == 80:
                hosts.update(LOOPBACK_NAMES)
            host = self.headers.get("Host", "")
            if host.lower() not in hosts:
                raise Forbidden(f"unexpected Host header {printable(host)!r}")
            origin = self.headers.get("Origin")
            if origin is not None and origin.lower() not in {f"http://{h}" for h in hosts}:
                raise Forbidden(f"cross-origin request from {printable(origin)!r} refused")

        def _guarded(self, route: Callable[[str], None]) -> None:
            """Every failure becomes a JSON error response instead of a dropped connection."""
            token = request_id.set(f"r{next(_request_ids):06d}")
            start, self._status = time.perf_counter(), 0
            path = urlsplit(self.path).path
            shown = printable(path)  # the raw path may carry newlines or terminal escapes
            try:
                self._check_origin()
                route(path)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                log.debug("client disconnected during %s %s", self.command, shown)
                return
            except ValidationError as e:  # includes BadRequest and its subclasses
                log.info("rejected %s %s: %s", self.command, shown, e)
                self._error(getattr(e, "status", HTTPStatus.BAD_REQUEST), str(e))
            except EmbeddingError as e:
                log.error("embedding backend unavailable: %s", e)
                self._error(HTTPStatus.SERVICE_UNAVAILABLE, f"scoring unavailable: {e}")
            except NoveltyError as e:
                log.exception("request failed: %s", e)
                self._error(HTTPStatus.INTERNAL_SERVER_ERROR, str(e))
            except OSError:  # disk or network trouble outside the embedder; its text names local paths
                log.exception("request failed")
                self._error(HTTPStatus.SERVICE_UNAVAILABLE,
                            f"service unavailable (request {request_id.get()}), see server log")
            except Exception:
                log.exception("unexpected error in %s %s", self.command, shown)
                self._error(HTTPStatus.INTERNAL_SERVER_ERROR,
                            f"internal error (request {request_id.get()}), see server log")
            finally:
                log.debug("%s %s -> %s in %.1f ms", self.command, shown, self._status,
                          (time.perf_counter() - start) * 1000)
                request_id.reset(token)

        def _state(self) -> dict:
            # Sizes only: shipping every headline on every response made payloads O(corpus).
            return {"corpus_size": len(app.scorer.corpus), "user_count": app.admitted_count}

        # -------------------------------------------------------------- routes

        def do_GET(self) -> None:
            self._guarded(self._get)

        def do_POST(self) -> None:
            self._guarded(self._post)

        def _not_allowed(self, path: str) -> None:
            self._error(HTTPStatus.METHOD_NOT_ALLOWED, f"{self.command} is not supported; use GET or POST")

        def _refuse(self) -> None:  # the stdlib would answer these with a 501 HTML page
            self._guarded(self._not_allowed)

        do_HEAD = do_PUT = do_DELETE = do_PATCH = do_OPTIONS = _refuse

        def _get(self, path: str) -> None:
            if path == "/":
                self._send(HTTPStatus.OK, (STATIC / "index.html").read_bytes(), "text/html; charset=utf-8")
            elif path == "/api/context":
                with app.lock:
                    self._json(HTTPStatus.OK, {
                        **self._state(),
                        "fixed": dataclasses.asdict(app.scorer.fixed),
                        "stances": [s.value for s in Stance],
                        "probes": app.probes,
                        "examples": app.examples,
                        "embedder": app.scorer.embedder.name,
                    })
            else:
                self._error(HTTPStatus.NOT_FOUND, "not found")

        def _post(self, path: str) -> None:
            if path == "/api/reset":
                self._read_json()  # content-type check: a cross-site form must not be able to wipe the log
                with app.lock:
                    app.reset()
                    self._json(HTTPStatus.OK, self._state())
                return
            if path != "/api/score":
                self._error(HTTPStatus.NOT_FOUND, "not found")
                return
            data = self._read_json()
            try:
                sub = Submission(headline=data.get("headline", ""), body=data.get("body", ""),
                                 stance=data.get("stance", ""))
            except ValidationError as e:
                raise BadRequest(str(e)) from None
            saved = None
            with app.lock:
                if data.get("commit") is True:
                    sub, result, saved = app.submit(sub)
                else:
                    result = app.scorer.score(sub)
                payload = dataclasses.asdict(result)
                for n in payload["nearest"]:
                    n["headline"] = app.headline(n["id"])
                response = {
                    **self._state(),
                    "result": payload,
                    "id": sub.id if result.admitted else None,
                }
            if saved is not None:
                response["saved"] = saved
            self._json(HTTPStatus.OK, response)

    return Handler


class _Server(ThreadingHTTPServer):
    """HTTPServer sets SO_REUSEADDR, which on Windows lets a second server bind a port that is
    already in use. Bind exclusively there, so "port in use" is reported instead of hidden."""

    allow_reuse_address = os.name != "nt"

    def server_bind(self) -> None:
        if os.name == "nt" and hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        super().server_bind()


class _IPv6Server(_Server):
    address_family = socket.AF_INET6


def _loopback_servers(host: str, port: int, handler) -> list[ThreadingHTTPServer]:
    """Listen on IPv4 *and* IPv6 loopback. On Windows "localhost" resolves to ::1 first, and a
    refused IPv6 connect costs ~200 ms per request before the client falls back to IPv4."""
    try:
        servers: list[ThreadingHTTPServer] = [_Server((host, port), handler)]
    except OSError as e:
        raise NoveltyError(f"cannot listen on {host}:{port}: {e.strerror or e} "
                           f"(is another server already running? try --port)") from e
    if host in ("127.0.0.1", "localhost") and socket.has_ipv6:
        try:
            servers.append(_IPv6Server(("::1", port), handler))
        except OSError as e:
            log.debug("IPv6 loopback unavailable (%s); IPv4 only", e)
    return servers


def _warmup(embedder) -> None:
    try:
        getattr(embedder, "warmup", lambda: None)()
    except EmbeddingError as e:
        log.warning("model warmup failed (%s); the first new text will retry", e)


def serve(port: int = 8000, host: str = "127.0.0.1", user_file: Path = USER_FILE) -> None:
    if not logging.getLogger("novelty").handlers:
        configure_logging()
    log.info("loading corpus and calibration...")
    app = App(build_scorer, user_file)
    servers = _loopback_servers(host, port, make_handler(app))
    # With a warm embedding cache the model was never loaded; load it now in the background so
    # the first new text does not pay for it, without delaying startup.
    threading.Thread(target=_warmup, args=(app.scorer.embedder,), daemon=True).start()
    log.info("%d saved user submission(s) in %s", app.admitted_count, user_file.name)
    log.info("open http://%s:%d  (embedder: %s, Ctrl+C to stop)", host, port, app.scorer.embedder.name)
    for extra in servers[1:]:
        threading.Thread(target=extra.serve_forever, daemon=True).start()
    try:
        servers[0].serve_forever()
    except KeyboardInterrupt:
        log.info("shutting down")
    finally:
        for srv in servers:
            srv.server_close()
