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
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np

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
    def __init__(self, model: str = "BAAI/bge-small-en-v1.5") -> None:
        self.name = f"fastembed:{model}"
        self._model_name = model
        self._model = None
        self._lock = threading.Lock()

    def warmup(self) -> None:
        with self._lock:
            if self._model is None:
                os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
                from fastembed import TextEmbedding  # heavy import (~0.6 s): only when needed

                self._model = TextEmbedding(self._model_name, cache_dir=str(CACHE_DIR / "fastembed"))

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        if self._model is None:
            self.warmup()
        return _normalize(np.array(list(self._model.embed(list(texts)))))


class GeminiEmbedder:
    BATCH = 100

    def __init__(self, model: str = "gemini-embedding-001", dimensions: int = 768) -> None:
        from google import genai

        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not set")
        self.name = f"gemini:{model}:{dimensions}"
        self._client = genai.Client(api_key=api_key)
        self._model = model
        self._dimensions = dimensions

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        from google.genai import types

        config = types.EmbedContentConfig(
            task_type="SEMANTIC_SIMILARITY", output_dimensionality=self._dimensions
        )
        vectors: list[list[float]] = []
        for i in range(0, len(texts), self.BATCH):
            resp = self._client.models.embed_content(
                model=self._model, contents=list(texts[i : i + self.BATCH]), config=config
            )
            vectors.extend(e.values for e in resp.embeddings)
        return _normalize(np.array(vectors))


class CachedEmbedder:
    _SQL_BATCH = 500  # stay well under SQLite's bound-parameter limit

    def __init__(self, inner: Embedder, cache_dir: Path = CACHE_DIR / "embeddings") -> None:
        self.inner = inner
        self.name = inner.name
        slug = hashlib.sha256(inner.name.encode()).hexdigest()[:12]
        cache_dir.mkdir(parents=True, exist_ok=True)
        self._path = cache_dir / f"{slug}.sqlite"
        self._hot: dict[str, np.ndarray] = {}  # keys seen this process: no SQLite round-trip
        self._lock = threading.Lock()
        self._db = self._open()
        self._import_legacy_json(cache_dir / f"{slug}.json")

    def _open(self) -> sqlite3.Connection:
        for attempt in range(2):
            try:
                db = sqlite3.connect(self._path, check_same_thread=False, isolation_level=None)
                db.execute("PRAGMA journal_mode=WAL")
                db.execute("PRAGMA synchronous=NORMAL")
                db.execute("CREATE TABLE IF NOT EXISTS vec (key TEXT PRIMARY KEY, data BLOB NOT NULL)")
                db.execute("SELECT 1 FROM vec LIMIT 1").fetchall()
                return db
            except sqlite3.DatabaseError:
                if attempt:
                    raise
                # a corrupt cache is only a cache: set it aside and start fresh
                os.replace(self._path, self._path.with_name(self._path.name + ".corrupt"))
        raise AssertionError("unreachable")

    def _import_legacy_json(self, legacy: Path) -> None:
        if not legacy.exists():
            return
        try:
            data = json.loads(legacy.read_text(encoding="utf-8"))
            rows = [(k, np.asarray(v, dtype=np.float32).tobytes()) for k, v in data.items()]
            with self._lock:
                self._db.executemany("INSERT OR IGNORE INTO vec VALUES (?, ?)", rows)
        except (ValueError, UnicodeDecodeError):
            pass
        legacy.unlink(missing_ok=True)

    @staticmethod
    def _key(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def warmup(self) -> None:
        if hasattr(self.inner, "warmup"):
            self.inner.warmup()

    def __len__(self) -> int:
        with self._lock:
            return self._db.execute("SELECT COUNT(*) FROM vec").fetchone()[0]

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        keys = [self._key(t) for t in texts]
        cold = [k for k in dict.fromkeys(keys) if k not in self._hot]
        with self._lock:
            for i in range(0, len(cold), self._SQL_BATCH):
                chunk = cold[i : i + self._SQL_BATCH]
                marks = ",".join("?" * len(chunk))
                for k, blob in self._db.execute(f"SELECT key, data FROM vec WHERE key IN ({marks})", chunk):
                    self._hot[k] = np.frombuffer(blob, dtype=np.float32)
        missing = {k: t for k, t in zip(keys, texts) if k not in self._hot}
        if missing:
            vecs = _validated(self.inner.embed(list(missing.values())), len(missing), self.name)
            with self._lock:
                self._db.execute("BEGIN")
                self._db.executemany("INSERT OR REPLACE INTO vec VALUES (?, ?)",
                                     [(k, v.tobytes()) for k, v in zip(missing, vecs)])
                self._db.execute("COMMIT")
            for k, v in zip(missing, vecs):
                self._hot[k] = v
        if not keys:
            return np.zeros((0, 0), dtype=np.float32)
        return np.stack([self._hot[k] for k in keys])


def _validated(vecs: np.ndarray, n: int, name: str) -> np.ndarray:
    """Refuse to cache (and score with) malformed vectors from a misbehaving backend."""
    vecs = np.asarray(vecs, dtype=np.float32)
    if vecs.ndim != 2 or vecs.shape[0] != n:
        raise RuntimeError(f"{name} returned {vecs.shape} vectors for {n} texts")
    if not np.isfinite(vecs).all():
        raise RuntimeError(f"{name} returned non-finite vector values")
    return vecs


def default_embedder() -> Embedder:
    """``NOVELTY_EMBEDDER`` = local | gemini. Defaults to gemini when a key is present."""
    choice = os.environ.get("NOVELTY_EMBEDDER")
    if choice is None:
        has_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        choice = "gemini" if has_key else "local"
    inner: Embedder = GeminiEmbedder() if choice == "gemini" else LocalEmbedder()
    return CachedEmbedder(inner)
