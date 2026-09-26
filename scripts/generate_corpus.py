"""Generate a synthetic corpus of submissions with Gemini.

    GEMINI_API_KEY=... python scripts/generate_corpus.py [--n 50] [--model MODEL] [--out data/corpus.generated.json] [--force]

Real comment sections are clustered: most people repeat a handful of obvious takes, a few say
something new. The prompt asks for that shape explicitly, because a corpus of 50 all-unique
comments would make "novel" meaningless. Writes to a separate file by default so the committed
corpus (which the tests are calibrated against) is not overwritten, and refuses to replace an
existing output file unless --force is given. --model defaults to $GEMINI_MODEL, else gemini-2.5-flash.

Exit codes: 0 ok, 1 generation failed (API, response or validation), 2 bad usage/configuration.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from novelty.data import DATA_DIR, load_fixed_content, write_json_atomic  # noqa: E402
from novelty.embeddings import gemini_api_key, is_transient_error  # noqa: E402
from novelty.errors import NoveltyError, ValidationError  # noqa: E402
from novelty.logging_setup import configure_logging  # noqa: E402
from novelty.models import Stance, Submission  # noqa: E402

log = logging.getLogger("novelty.generate_corpus")

PROMPT = """You are simulating the comment section of a local news site.

Article:
\"\"\"{article}\"\"\"

Write {n} distinct reader submissions. Each has:
- headline: under 80 characters
- body: 1-3 sentences, 20-60 words, plain conversational English
- stance: one of support, oppose, mixed, undecided

Make the distribution realistic, like a real comment section:
- About 80% should repeat the few most obvious reactions to this article, in different words
  (e.g. worries about lost parking, praise for green space, complaints about cost, doubts about
  the shuttle, heat and trees, safety and upkeep). Several people should make essentially the
  same point.
- About 20% should be less common but still clearly about the article.
- Do not invent specific facts that contradict the article.
- Vary voice: shop owners, parents, commuters, retirees, renters."""

SCHEMA = {
    "type": "ARRAY",
    "items": {
        "type": "OBJECT",
        "properties": {
            "headline": {"type": "STRING"},
            "body": {"type": "STRING"},
            "stance": {"type": "STRING", "enum": [s.value for s in Stance]},
        },
        "required": ["headline", "body", "stance"],
    },
}
ATTEMPTS = 3


class GenerationError(NoveltyError):
    pass


def _generate(client, types, model: str, prompt: str) -> str:
    config = types.GenerateContentConfig(response_mime_type="application/json", response_schema=SCHEMA, temperature=1.0)
    for attempt in range(1, ATTEMPTS + 1):
        try:
            resp = client.models.generate_content(model=model, contents=prompt, config=config)
            if not getattr(resp, "text", None):
                raise GenerationError("Gemini returned an empty response (blocked or truncated?)")
            return resp.text
        except GenerationError:
            raise
        except Exception as e:
            if attempt == ATTEMPTS or not is_transient_error(e):
                raise GenerationError(f"Gemini request failed: {e}") from e
            delay = 2 ** attempt
            log.warning("attempt %d/%d failed (%s); retrying in %d s", attempt, ATTEMPTS, e, delay)
            time.sleep(delay)
    raise AssertionError("unreachable")


def _rows(text: str) -> list[dict]:
    try:
        items = json.loads(text)
    except json.JSONDecodeError as e:
        raise GenerationError(f"Gemini response is not valid JSON: {e}") from e
    if not isinstance(items, list):
        raise GenerationError(f"expected a JSON list of submissions, got {type(items).__name__}")
    rows = []
    for i, item in enumerate(items, start=1):
        try:
            sub = Submission.from_dict(item)  # validates the three properties
        except ValidationError as e:
            log.warning("skipping invalid item %d: %s", i, e)
            continue
        rows.append({"id": f"g{i:02d}", "stance": sub.stance.value, "headline": sub.headline, "body": sub.body})
    return rows


def _write(path: Path, rows: list[dict]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json_atomic(path, rows)
    except OSError as e:
        raise GenerationError(f"could not write {path}: {e}") from e


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--model", default=os.environ.get("GEMINI_MODEL", "gemini-2.5-flash"))
    parser.add_argument("--out", type=Path, default=DATA_DIR / "corpus.generated.json")
    parser.add_argument("--force", action="store_true", help="overwrite --out if it already exists")
    args = parser.parse_args(argv)
    configure_logging()

    if not 1 <= args.n <= 500:
        log.error("--n must be between 1 and 500")
        return 2
    if args.out.exists() and not args.force:
        log.error("%s already exists; pass --force to overwrite it", args.out)
        return 2
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        log.error('google-genai is not installed; run: pip install -e ".[gemini]"')
        return 2
    api_key = gemini_api_key()
    if not api_key:
        log.error("set GEMINI_API_KEY (free key: https://aistudio.google.com/)")
        return 2

    try:
        fixed = load_fixed_content()
        log.info("asking %s for %d submissions", args.model, args.n)
        text = _generate(genai.Client(api_key=api_key), types, args.model, PROMPT.format(article=fixed.text, n=args.n))
        rows = _rows(text)
        if not rows:
            raise GenerationError("no valid submissions in the response")
        if len(rows) < args.n:
            log.warning("only %d of %d requested submissions were valid", len(rows), args.n)
        _write(args.out, rows)
    except NoveltyError as e:
        log.error("%s", e)
        return 1
    log.info("wrote %d submissions to %s", len(rows), args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
