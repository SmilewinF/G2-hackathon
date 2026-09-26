"""Loaders for the bundled fixed content, corpus, calibration anchors, test probes and the web
UI's saved user submissions.

Every loader raises ``DataError`` naming the file (and the item, for lists) when a file is
missing, is not valid JSON, or has the wrong shape, instead of a bare ``KeyError`` deep inside.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from .embeddings import Embedder, default_embedder
from .errors import DataError, ValidationError
from .models import FixedContent, Stance, Submission
from .scorer import NoveltyScorer, ScorerConfig

log = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
USER_FILE = DATA_DIR / "user_submissions.json"  # written by the web UI, gitignored


def _load(name: str, expected: type) -> Any:
    path = DATA_DIR / name
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise DataError(f"data file not found: {path}") from None
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise DataError(f"{name} is not valid JSON: {e}") from None
    except OSError as e:
        raise DataError(f"cannot read {path}: {e}") from None
    if not isinstance(data, expected):
        raise DataError(f"{name} must contain a JSON {expected.__name__}, got {type(data).__name__}")
    return data


def _item_id(item: Any) -> str:
    return repr(item.get("id")) if isinstance(item, dict) else "not an object"


def _submissions(items: list, source: str) -> list[Submission]:
    subs, seen = [], set()
    for i, item in enumerate(items):
        try:
            sub = Submission.from_dict(item)
        except ValidationError as e:
            raise DataError(f"{source} item {i} ({_item_id(item)}): {e}") from None
        if sub.id is not None:
            if sub.id in seen:
                raise DataError(f"{source} item {i}: duplicate id {sub.id!r}")
            seen.add(sub.id)
        subs.append(sub)
    return subs


def load_fixed_content() -> FixedContent:
    d = _load("fixed_content.json", dict)
    missing = [k for k in ("id", "title", "text") if k not in d]
    if missing:
        raise DataError(f"fixed_content.json is missing field(s): {', '.join(missing)}")
    try:
        return FixedContent(id=d["id"], title=d["title"], text=d["text"])
    except ValidationError as e:
        raise DataError(f"fixed_content.json: {e}") from None


def load_corpus() -> list[Submission]:
    return _submissions(_load("corpus.json", list), "corpus.json")


def load_off_topic_anchors() -> list[str]:
    anchors = _load("off_topic_anchors.json", list)
    bad = [i for i, a in enumerate(anchors) if not isinstance(a, str) or not a.strip()]
    if bad or not anchors:
        raise DataError(f"off_topic_anchors.json must be a non-empty list of strings (bad items: {bad})")
    return anchors


def load_probes() -> dict[str, list[dict]]:
    """Labelled probe submissions grouped by expected behaviour (see data/probes.json)."""
    probes = {k: v for k, v in _load("probes.json", dict).items() if not k.startswith("_")}
    for group, items in probes.items():
        if not isinstance(items, list):
            raise DataError(f"probes.json group {group!r} must be a list")
        _submissions(items, f"probes.json[{group}]")  # validate every probe
    return probes


EXAMPLE_EXPECTATIONS = ("novel", "common", "off_topic")


def load_examples() -> dict[str, list[dict]]:
    """The web UI's example submissions (data/examples.json): exactly two per stance, each with
    the outcome it demonstrates ("expect") and a short display label."""
    raw = {k: v for k, v in _load("examples.json", dict).items() if not k.startswith("_")}
    stances = [s.value for s in Stance]
    if sorted(raw) != sorted(stances):
        raise DataError(f"examples.json must have exactly the stances {stances}, got {sorted(raw)}")
    examples = {}
    for stance in stances:  # stance order, not file order
        items = raw[stance]
        if not isinstance(items, list) or len(items) != 2:
            raise DataError(f"examples.json[{stance}] must be a list of exactly 2 examples")
        for i, item in enumerate(items):
            if not isinstance(item, dict) or item.get("expect") not in EXAMPLE_EXPECTATIONS or not item.get("label"):
                raise DataError(f"examples.json[{stance}] item {i} needs a label and expect in {EXAMPLE_EXPECTATIONS}")
        _submissions([{**item, "stance": stance} for item in items], f"examples.json[{stance}]")
        examples[stance] = [{**item, "stance": stance} for item in items]
    return examples


def load_user_submissions(path: Path = USER_FILE, admitted_only: bool = True) -> list[Submission]:
    """Submissions saved by the web UI. Read-only: unlike the server, it never moves the file.

    Invalid records are skipped with a warning, so one hand-edited record cannot hide the rest.
    """
    if not path.exists():
        return []
    try:
        records = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError, OSError) as e:
        raise DataError(f"{path.name} is unreadable: {e}") from None
    if not isinstance(records, list):
        raise DataError(f"{path.name} must contain a JSON list")
    subs, seen = [], set()
    for i, rec in enumerate(records):
        if not isinstance(rec, dict) or (admitted_only and not rec.get("admitted")):
            continue
        try:
            sub = Submission.from_dict(rec)
        except ValidationError as e:
            log.warning("skipping %s item %d (%s): %s", path.name, i, _item_id(rec), e)
            continue
        if sub.id in seen:
            log.warning("skipping %s item %d: duplicate id %r", path.name, i, sub.id)
            continue
        seen.add(sub.id)
        subs.append(sub)
    return subs


def build_scorer(
    embedder: Embedder | None = None,
    config: ScorerConfig = ScorerConfig(),
    include_user_submissions: bool = False,
    user_file: Path = USER_FILE,
) -> NoveltyScorer:
    """Scorer over the seed corpus, optionally plus the web UI's admitted submissions."""
    corpus = load_corpus()
    if include_user_submissions:
        seed_ids = {s.id for s in corpus}
        extra = [s for s in load_user_submissions(user_file) if s.id not in seed_ids]
        if extra:
            log.info("including %d user submission(s) from %s", len(extra), user_file.name)
        corpus += extra
    return NoveltyScorer(
        fixed=load_fixed_content(),
        corpus=corpus,
        embedder=embedder or default_embedder(),
        off_topic_anchors=load_off_topic_anchors(),
        config=config,
    )
