"""Generate a synthetic corpus of submissions with Gemini.

    GEMINI_API_KEY=... python scripts/generate_corpus.py [--n 50] [--out data/corpus.generated.json]

Real comment sections are clustered: most people repeat a handful of obvious takes, a few say
something new. The prompt asks for that shape explicitly, because a corpus of 50 all-unique
comments would make "novel" meaningless. Writes to a separate file by default so the committed
corpus (which the tests are calibrated against) is not overwritten.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from novelty.data import DATA_DIR, load_fixed_content  # noqa: E402
from novelty.models import Stance, Submission  # noqa: E402

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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--model", default=os.environ.get("GEMINI_MODEL", "gemini-2.5-flash"))
    parser.add_argument("--out", type=Path, default=DATA_DIR / "corpus.generated.json")
    args = parser.parse_args()

    from google import genai
    from google.genai import types

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        sys.exit("Set GEMINI_API_KEY (free key: https://aistudio.google.com/)")

    schema = {
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
    fixed = load_fixed_content()
    resp = genai.Client(api_key=api_key).models.generate_content(
        model=args.model,
        contents=PROMPT.format(article=fixed.text, n=args.n),
        config=types.GenerateContentConfig(
            response_mime_type="application/json", response_schema=schema, temperature=1.0
        ),
    )

    rows = []
    for i, item in enumerate(json.loads(resp.text), start=1):
        try:
            sub = Submission.from_dict(item)  # validates the three properties
        except ValueError as e:
            print(f"skipping invalid item {i}: {e}", file=sys.stderr)
            continue
        rows.append({"id": f"g{i:02d}", "stance": sub.stance.value, "headline": sub.headline, "body": sub.body})

    args.out.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(rows)} submissions to {args.out}")


if __name__ == "__main__":
    main()
