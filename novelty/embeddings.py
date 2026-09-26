"""Embedding backends.

- ``LocalEmbedder``: BAAI/bge-small-en-v1.5 via fastembed (ONNX, CPU, no API key). Default.
- ``GeminiEmbedder``: Gemini ``gemini-embedding-001`` with task type SEMANTIC_SIMILARITY.

Both return L2-normalised float32 vectors, so a dot product is cosine similarity.
``CachedEmbedder`` wraps either one with a JSON disk cache keyed by model + text hash.
"""

from __future__ import annotations

import hashlib
import json
import os
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
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
        from fastembed import TextEmbedding

        self.name = f"fastembed:{model}"
        self._model = TextEmbedding(model, cache_dir=str(CACHE_DIR / "fastembed"))

    def embed(self, texts: Sequence[str]) -> np.ndarray:
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
    def __init__(self, inner: Embedder, cache_dir: Path = CACHE_DIR / "embeddings") -> None:
        self.inner = inner
        self.name = inner.name
        slug = hashlib.sha256(inner.name.encode()).hexdigest()[:12]
        self._path = cache_dir / f"{slug}.json"
        self._cache: dict[str, list[float]] = {}
        if self._path.exists():
            try:
                self._cache = json.loads(self._path.read_text(encoding="utf-8"))
            except (ValueError, UnicodeDecodeError):
                self._cache = {}  # a corrupt cache is only a cache: rebuild it

    @staticmethod
    def _key(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        keys = [self._key(t) for t in texts]
        missing = list(dict.fromkeys(t for t, k in zip(texts, keys) if k not in self._cache))
        if missing:
            vecs = _validated(self.inner.embed(missing), len(missing), self.name)
            for text, vec in zip(missing, vecs):
                self._cache[self._key(text)] = vec.tolist()
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._path.with_name(self._path.name + ".tmp")
            tmp.write_text(json.dumps(self._cache), encoding="utf-8")
            os.replace(tmp, self._path)  # atomic, so an interrupted run cannot corrupt the cache
        return np.array([self._cache[k] for k in keys], dtype=np.float32).reshape(len(keys), -1)


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
