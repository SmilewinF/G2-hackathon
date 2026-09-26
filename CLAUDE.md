# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Hackathon solution (G2, "Rewarding novelty in submissions"): score a user submission's novelty against ~50 others on [0, 1], while giving zero reward to submissions that are novel only because they're off-topic. The README is the submission write-up. COLLABORATION.md documents how the agent was prompted, as the brief requires, so update it when a design decision changes.

## Commands

```bash
.venv/Scripts/python -m pip install -e ".[dev,gemini]"   # venv lives in .venv (Windows layout)
.venv/Scripts/python -m pytest                            # full suite, ~2 s
.venv/Scripts/python -m pytest tests/test_edge_cases.py -k typo
.venv/Scripts/python -m novelty serve                     # web UI at http://127.0.0.1:8000 (stdlib only)
.venv/Scripts/python -m novelty demo                      # score table for data/probes.json — check after any scoring change
.venv/Scripts/python -m novelty corpus                    # leave-one-out novelty per corpus item
.venv/Scripts/python -m novelty score --headline ... --body ... --stance support [--json]
```

The first run downloads the fastembed ONNX model into `.cache/fastembed`. Embeddings are cached in `.cache/embeddings/<model-hash>.sqlite`, keyed by a hash of the *prepared* text. Delete that folder only if you change an embedder's output for the same text.

## Architecture

Pipeline: `Submission` (normalised on construction) → `TextPreparer` → `ReferenceIndex.analyze` → signals → combine → admission policy.

- **`preparation.py`**: per sentence, it detects the language from function words and script, expands slang, and applies single-edit spelling correction that prefers article and corpus words. Foreign sentences stay untouched and are excluded via `PreparedText.analysis_text` and the `foreign` flags. Typos used to read as novelty, so this runs on corpus entries and queries alike.
- **`index.py`**: `ReferenceIndex` holds the article (id `"article"`, always a reference item) plus admitted submissions. Each text is prepared, tokenised and embedded once, in one batch per call. Dense rows use growable matrices. Lexical vectors are sparse, stored in `lexical.SparseVectors` (an inverted index). `index.version` changes only when `TfidfModel` refits, which happens after 10% corpus growth.
- **`signals/`**: each signal implements `fit(index)` and `evaluate(analysis, index)`, and can override `update(index, added)` for incremental recalibration. The default for `update` is a full `fit`. They combine as `novelty = min(NOVELTY) × Π(MODIFIER)` and `score = novelty × Π(RELEVANCE)`. A new component is a new `Signal` subclass passed via `NoveltyScorer(signals=[...])`.
  - `WholeTextNovelty`: nearest and top-k hybrid distance, calibrated with a leave-one-out robust z → normal CDF. It keeps a per-entry top-k.
  - `ClauseCoverage`: the most novel relevant clause, applied only with ≥ 2 substantive clauses. It keeps each clause's best match.
  - `DuplicateCheck`: shingle containment, but only against the 25 most similar entries plus the article.
  - `ContentQuality`, `Specificity`, `StanceRarity`.
  - `TopicMargin`: contrastive margin of the *substantive body* against the generic-chatter centroid (`data/off_topic_anchors.json`). A margin ≤ 0 means off-topic.
- **`scorer.py`**: orchestration. `submit()` adds a submission only if it's relevant, non-duplicate and substantive. `add_many()` recalibrates once per batch. `fork()` deep-copies the state while sharing the embedder and preparer, which tests and the server's Reset rely on.
- **Invariant:** incremental `update()` must equal a full `fit()` on the same index. `test_incremental_calibration_matches_a_full_refit` guards this, so extend it when you add a stateful signal.

Embedding backends are in `novelty/embeddings.py`. `NOVELTY_EMBEDDER=local|gemini` selects one; the default is gemini if `GEMINI_API_KEY` is set, otherwise local bge-small, lazy-loaded. Tests force local through `tests/conftest.py`, and `NOVELTY_TEST_EMBEDDER=gemini` overrides that.

The web UI (`novelty/server.py`) logs every committed attempt to `data/user_submissions.json` (gitignored) and replays the admitted ones with `add_many` on startup. Never write them into `corpus.json`: the tests depend on that seed set. The server also listens on `::1`, because Windows resolves `localhost` to IPv6 first and a refused connect cost 200 ms per request.

## Data and test conventions

- `data/probes.json` holds the labelled test inputs, grouped as `novel_relevant`, `duplicates` and `off_topic`. The behaviour tests index into these groups by position, so add new probes at the end of a group. Novel and off-topic probes must not appear in `corpus.json` or `off_topic_anchors.json` (`test_shape.py` enforces this).
- Test thresholds: novel ≥ 0.6, non-novel ≤ 0.25, off-topic ≤ 0.01. Some tests refer to corpus ids directly (`c01`, `c11`, `c27`, and the crowded/one-off lists), so editing `corpus.json` can break them.
- `test_adversarial.py` and `test_edge_cases.py` pin behaviours that were once broken; each comment says what the case scored before its fix. Don't loosen them to make a change pass.
- The `scorer` fixture is a fork of one session-scoped build. Use `base_scorer.fork` wherever a test needs a scorer factory.
- Known weak spots: short question-style comments (`c36`) get only partial relevance, and code-mixed sentences earn nothing. See the README limitations section before "fixing" the gate thresholds.

## Workflow

Commit after each feature or change and push to the user's GitHub remote. That's a standing instruction; `.venv/` and `.cache/` are gitignored. Measure before optimising: the README's Performance table records what was measured and what was deliberately left alone.
