"""Embedding backends.

- ``LocalEmbedder``: BAAI/bge-small-en-v1.5 via fastembed (ONNX, CPU, no API key). Default.
- ``GeminiEmbedder``: Gemini ``gemini-embedding-001`` with task type SEMANTIC_SIMILARITY.

Both return L2-normalised float32 vectors, so a dot product is cosine similarity.

``CachedEmbedder`` wraps either one with a SQLite cache keyed by model + text hash. The earlier
JSON cache rewrote the whole file on every miss, which was ~77 % of request time and grew with
every request; SQLite writes one row per new text, stores float32 blobs (~4x smaller than JSON)
and is safe against interrupted writes.

Model loading is lazy: with a warm cache, starting the scorer never loads the ONNX model (~1 s);
``warmup()`` loads it ahead of the first cache miss, e.g. in a background thread.

Failures: backend problems raise ``EmbeddingError``. The cache is only an optimisation, so a
cache that cannot be read or written is logged and bypassed rather than failing the request.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
import threading
import time
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np

from .errors import EmbeddingError

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = Path(os.environ.get("NOVELTY_CACHE_DIR", PROJECT_ROOT / ".cache"))


class Embedder(Protocol):
    name: str

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Return an (n, d) array of unit-length vectors."""
        ...


def _normalize(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, dtype=np.float32)
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    return m / np.clip(norms, 1e-12, None)


class LocalEmbedder:
    # fastembed pads every text in a batch to the batch's longest one. A submission is embedded as
    # its whole text (up to ~450 tokens) plus each clause (~5-15), so one batch made every clause
    # cost as much as the whole comment: 64 clauses took 6.2 s. Sorted by length in batches of 16,
    # the same texts take 0.22 s and give bit-identical vectors.
    BATCH = 16

    def __init__(self, model: str = "BAAI/bge-small-en-v1.5") -> None:
        self.name = f"fastembed:{model}"
        self._model_name = model
        self._model = None
        self._lock = threading.Lock()

    def warmup(self) -> None:
        with self._lock:
            if self._model is not None:
                return
            os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
            start = time.perf_counter()
            try:
                from fastembed import TextEmbedding  # heavy import (~0.6 s): only when needed

                self._model = TextEmbedding(self._model_name, cache_dir=str(CACHE_DIR / "fastembed"))
            except ImportError as e:
                raise EmbeddingError(f"fastembed is not installed ({e}); run: pip install -e .") from e
            except Exception as e:  # download, disk or ONNX runtime failure
                raise EmbeddingError(
                    f"could not load embedding model {self._model_name}: {e} (the first run downloads "
                    f"~70 MB into {CACHE_DIR / 'fastembed'}; check network access and disk space)"
                ) from e
            log.info("loaded embedding model %s in %.1f s", self._model_name, time.perf_counter() - start)

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        if self._model is None:
            self.warmup()
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        try:
            vecs = np.array(list(self._model.embed([texts[i] for i in order], batch_size=self.BATCH)))
        except Exception as e:
            raise EmbeddingError(f"{self.name} failed to embed {len(texts)} text(s): {e}") from e
        out = np.empty_like(vecs)
        out[order] = vecs
        return _normalize(out)


def _network_errors() -> tuple[type[Exception], ...]:
    """httpx's timeout and connection errors (google-genai's transport); they do not derive from
    the built-in TimeoutError/ConnectionError, so they were never retried."""
    try:
        import httpx
    except ImportError:
        return ()
    return (httpx.TimeoutException, httpx.NetworkError, httpx.RemoteProtocolError)


def is_transient_error(e: Exception) -> bool:
    """Worth retrying: rate limits, server errors, timeouts, dropped connections."""
    code = getattr(e, "code", None) or getattr(e, "status_code", None)
    return code in (408, 429, 500, 502, 503, 504) or isinstance(e, (TimeoutError, ConnectionError, *_network_errors()))


def gemini_api_key() -> str | None:
    """The Gemini key from ``GEMINI_API_KEY``, falling back to ``GOOGLE_API_KEY``."""
    return os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")


class GeminiEmbedder:
    BATCH = 100
    ATTEMPTS = 3
    TIMEOUT_MS = 30_000  # per request; google-genai's default is no timeout, and the server scores under a lock

    def __init__(self, model: str = "gemini-embedding-001", dimensions: int = 768) -> None:
        try:
            from google import genai
            from google.genai import types
        except ImportError as e:
            raise EmbeddingError(f"google-genai is not installed ({e}); run: pip install -e \".[gemini]\"") from e

        api_key = gemini_api_key()
        if not api_key:
            raise EmbeddingError("GEMINI_API_KEY is not set (free key: https://aistudio.google.com/)")
        self.name = f"gemini:{model}:{dimensions}"
        self._client = genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=self.TIMEOUT_MS))
        self._model = model
        self._dimensions = dimensions

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        from google.genai import types

        config = types.EmbedContentConfig(
            task_type="SEMANTIC_SIMILARITY", output_dimensionality=self._dimensions
        )
        vectors: list[list[float]] = []
        # The API rejects empty content; a body of only slang ("lol lol") prepares to blank text.
        texts = [t if t.strip() else "." for t in texts]
        for i in range(0, len(texts), self.BATCH):
            resp = self._call(list(texts[i : i + self.BATCH]), config)
            vectors.extend(e.values for e in resp.embeddings)
        return _normalize(np.array(vectors))

    def _call(self, batch: list[str], config):
        for attempt in range(1, self.ATTEMPTS + 1):
            try:
                return self._client.models.embed_content(model=self._model, contents=batch, config=config)
            except Exception as e:
                if attempt == self.ATTEMPTS or not is_transient_error(e):
                    raise EmbeddingError(f"Gemini embedding request failed: {e}") from e
                delay = 2 ** (attempt - 1)
                log.warning("Gemini embedding attempt %d/%d failed (%s); retrying in %d s",
                            attempt, self.ATTEMPTS, e, delay)
                time.sleep(delay)
        raise AssertionError("unreachable")


class CachedEmbedder:
    _SQL_BATCH = 500  # stay well under SQLite's bound-parameter limit

    def __init__(self, inner: Embedder, cache_dir: Path = CACHE_DIR / "embeddings") -> None:
        self.inner = inner
        self.name = inner.name
        slug = hashlib.sha256(inner.name.encode()).hexdigest()[:12]
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            log.warning("cannot create cache directory %s (%s); embeddings will not be cached on disk", cache_dir, e)
        self._path = cache_dir / f"{slug}.sqlite"
        self._hot: dict[str, np.ndarray] = {}  # keys seen this process: no SQLite round-trip
        self._lock = threading.Lock()
        self.hits = self.misses = 0  # texts served from cache / sent to the backend
        self.embed_seconds = 0.0  # time spent in the backend
        self._db: sqlite3.Connection | None = self._open()
        self._import_legacy_json(cache_dir / f"{slug}.json")

    def _open(self) -> sqlite3.Connection | None:
        for attempt in range(2):
            db = None
            try:
                db = sqlite3.connect(self._path, check_same_thread=False, isolation_level=None)
                db.execute("PRAGMA journal_mode=WAL")
                db.execute("PRAGMA synchronous=NORMAL")
                db.execute("CREATE TABLE IF NOT EXISTS vec (key TEXT PRIMARY KEY, data BLOB NOT NULL)")
                db.execute("SELECT 1 FROM vec LIMIT 1").fetchall()
                return db
            except sqlite3.DatabaseError as e:
                if db is not None:
                    db.close()  # a connection to a corrupt file must be closed before it can be moved aside
                if attempt:
                    log.warning("embedding cache %s unusable (%s); continuing without a disk cache", self._path, e)
                    return None
                # a corrupt cache is only a cache: set it aside and start fresh
                log.warning("embedding cache %s is corrupt (%s); moving it aside", self._path.name, e)
                try:
                    os.replace(self._path, self._path.with_name(self._path.name + ".corrupt"))
                except OSError as move_error:
                    log.warning("could not move the corrupt cache aside: %s", move_error)
                    return None
        raise AssertionError("unreachable")

    def _import_legacy_json(self, legacy: Path) -> None:
        if not legacy.exists() or self._db is None:
            return
        try:
            data = json.loads(legacy.read_text(encoding="utf-8"))
            rows = [(k, np.asarray(v, dtype=np.float32).tobytes()) for k, v in data.items()]
            with self._lock:
                self._db.executemany("INSERT OR IGNORE INTO vec VALUES (?, ?)", rows)
            log.info("imported %d vectors from legacy cache %s", len(rows), legacy.name)
        except (ValueError, UnicodeDecodeError, AttributeError, sqlite3.Error) as e:
            log.warning("could not import legacy cache %s (%s); it will be rebuilt", legacy.name, e)
        try:
            legacy.unlink(missing_ok=True)
        except OSError as e:
            log.warning("could not delete legacy cache %s: %s", legacy.name, e)

    @staticmethod
    def _key(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def warmup(self) -> None:
        if hasattr(self.inner, "warmup"):
            self.inner.warmup()

    def close(self) -> None:
        """Close the cache database; embedding afterwards keeps working, from memory only."""
        with self._lock:
            if self._db is not None:
                self._db.close()
                self._db = None

    def __enter__(self) -> CachedEmbedder:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __len__(self) -> int:
        if self._db is None:
            return len(self._hot)
        with self._lock:
            return self._db.execute("SELECT COUNT(*) FROM vec").fetchone()[0]

    def _read(self, keys: list[str]) -> None:
        if self._db is None or not keys:
            return
        try:
            with self._lock:
                for i in range(0, len(keys), self._SQL_BATCH):
                    chunk = keys[i : i + self._SQL_BATCH]
                    marks = ",".join("?" * len(chunk))
                    for k, blob in self._db.execute(f"SELECT key, data FROM vec WHERE key IN ({marks})", chunk):
                        self._hot[k] = np.frombuffer(blob, dtype=np.float32)
        except sqlite3.Error as e:
            log.warning("embedding cache read failed (%s); treating as cache misses", e)

    def _write(self, rows: list[tuple[str, bytes]]) -> None:
        if self._db is None:
            return
        try:
            with self._lock:
                self._db.execute("BEGIN")
                try:
                    self._db.executemany("INSERT OR REPLACE INTO vec VALUES (?, ?)", rows)
                    self._db.execute("COMMIT")
                except sqlite3.Error:
                    self._db.execute("ROLLBACK")
                    raise
        except sqlite3.Error as e:
            log.warning("embedding cache write failed (%s); vectors are kept in memory only", e)

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        keys = [self._key(t) for t in texts]
        self._read([k for k in dict.fromkeys(keys) if k not in self._hot])
        missing = {k: t for k, t in zip(keys, texts) if k not in self._hot}
        self.hits += len(set(keys)) - len(missing)
        if missing:
            start = time.perf_counter()
            vecs = _validated(self.inner.embed(list(missing.values())), len(missing), self.name)
            self.embed_seconds += time.perf_counter() - start
            self.misses += len(missing)
            self._write([(k, v.tobytes()) for k, v in zip(missing, vecs)])
            for k, v in zip(missing, vecs):
                self._hot[k] = v
        if not keys:
            return np.zeros((0, 0), dtype=np.float32)
        return np.stack([self._hot[k] for k in keys])


def _validated(vecs: np.ndarray, n: int, name: str) -> np.ndarray:
    """Refuse to cache (and score with) malformed vectors from a misbehaving backend."""
    try:
        vecs = np.asarray(vecs, dtype=np.float32)
    except (TypeError, ValueError) as e:
        raise EmbeddingError(f"{name} returned vectors that are not numeric: {e}") from e
    if vecs.ndim != 2 or vecs.shape[0] != n:
        raise EmbeddingError(f"{name} returned {vecs.shape} vectors for {n} texts")
    if not np.isfinite(vecs).all():
        raise EmbeddingError(f"{name} returned non-finite vector values")
    return vecs


def default_embedder() -> Embedder:
    """``NOVELTY_EMBEDDER`` = local | gemini, wrapped in a ``CachedEmbedder``. Defaults to gemini when ``gemini_api_key()`` finds a key."""
    choice = os.environ.get("NOVELTY_EMBEDDER")
    if choice is None:
        choice = "gemini" if gemini_api_key() else "local"
        if choice == "gemini" and not os.environ.get("GEMINI_API_KEY"):
            # GOOGLE_API_KEY is a generic name other tools set too; say that text now goes to Google.
            log.info("using Gemini embeddings because GOOGLE_API_KEY is set (NOVELTY_EMBEDDER=local keeps text on this machine)")
    if choice not in ("local", "gemini"):
        raise EmbeddingError(f"NOVELTY_EMBEDDER must be 'local' or 'gemini', got {choice!r}")
    inner: Embedder = GeminiEmbedder() if choice == "gemini" else LocalEmbedder()
    return CachedEmbedder(inner)
