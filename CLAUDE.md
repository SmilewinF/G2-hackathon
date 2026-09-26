# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Hackathon solution (G2, "Rewarding novelty in submissions"): score a user submission's novelty against ~50 others on [0, 1], while giving zero reward to submissions that are novel only because they're off-topic. The README is the submission write-up. COLLABORATION.md documents how the agent was prompted, as the brief requires, so update it when a design decision changes.

## Commands

```bash
.venv/Scripts/python -m pip install -e ".[dev,gemini]"   # venv lives in .venv (Windows layout)
.venv/Scripts/python -m pytest                            # full suite, ~6 s, seed corpus (reproducible); test-split checks deselected
NOVELTY_TEST_CORPUS=full .venv/Scripts/python -m pytest   # + admitted web-UI submissions (live-corpus check; results depend on them)
.venv/Scripts/python -m pytest -m heldout_test            # the held-out TEST split checks: report only, never tune on them
.venv/Scripts/python -m pytest tests/test_edge_cases.py -k typo
.venv/Scripts/python -m novelty serve                     # web UI at http://127.0.0.1:8000 (stdlib only)
.venv/Scripts/python -m novelty eval                      # held-out evaluation, dev split (default); tune on dev ONLY, report --split test
.venv/Scripts/python -m novelty demo                      # score table for data/probes.json — check after any scoring change
.venv/Scripts/python -m novelty corpus                    # leave-one-out novelty per corpus item
.venv/Scripts/python -m novelty score --headline ... --body ... --stance support [--json]
```

The first run downloads the fastembed ONNX model into `.cache/fastembed`. Embeddings are cached in `.cache/embeddings/<model-hash>.sqlite`, keyed by a hash of the *prepared* text. Delete that folder only if you change an embedder's output for the same text. `NOVELTY_CACHE_DIR` relocates the whole cache (model + embeddings); point a worktree at the main checkout's `.cache` to skip the download.

## Architecture

Pipeline: `Submission` (normalised on construction) → `TextPreparer` → `ReferenceIndex.analyze` → signals → combine → admission policy.

- **`preparation.py`**: per sentence, it detects the language from function words and script (whole Unicode words, accents folded only to match the function-word list), expands slang, and applies single-edit spelling correction that prefers article and corpus words. Foreign sentences stay untouched. A clause is foreign if it looks foreign or lies in a foreign sentence (`text.clause_spans` gives positions); foreign sentences and clauses, and foreign headline sentences, are cut out of `PreparedText.analysis_text` (English-only text is left byte-identical, so its cache entries stay valid). Typos used to read as novelty, so this runs on corpus entries and queries alike.
- **`index.py`**: `ReferenceIndex` holds the article (id `"article"`, always a reference item) plus admitted submissions, and the content vectors of the generic anchors and the relevance negatives (`anchor_content`, `negative_content`). Each text is prepared, tokenised and embedded once, in one batch per call. Dense rows use growable matrices. Lexical vectors are sparse, stored in `lexical.SparseVectors` (an inverted index). `index.version` changes only when `TfidfModel` refits, which happens after 10% corpus growth.
- **`signals/`**: each signal implements `fit(index)` and `evaluate(analysis, index)`, and can override `update(index, added)` for incremental recalibration. The default for `update` is a full `fit`. They combine as `novelty = min(NOVELTY) × Π(MODIFIER)` and `score = novelty × Π(RELEVANCE)`. A new component is a new `Signal` subclass passed via `NoveltyScorer(signals=[...])`.
  - `WholeTextNovelty`: nearest and top-k hybrid distance, calibrated with a leave-one-out robust z → normal CDF. It keeps a per-entry top-k.
  - `ClauseCoverage`: the most novel on-topic clause, applied only with ≥ 2 substantive clauses. On-topic comes from `Analysis.clause_relevance`, which the scorer fills from the relevance signals' `clause_relevance()` (learned gate > 0.5; margin fallback: margin > 0); a clause also needs ≥ 2 specific words. It keeps each clause's best match and reports `on_topic_share` for admission.
  - `DuplicateCheck`: shingle containment, but only against the 25 most similar entries plus the article. A copy needs an entry ≥ 60% contained *and* the contained entries making up ≥ 50% of the text, so quoting a short comment is not a copy.
  - `ContentQuality`, `Specificity`, `StanceRarity`.
  - `TopicDiscriminant` (default relevance): a shrinkage Fisher discriminant on the *substantive body*, trained on the on-topic references (article + on-topic submissions) vs `data/relevance_negatives.json` + the generic anchors, boundary at the training equal-error point (the balanced-accuracy optimum jumped when ordinary comments were admitted), gate across boundary ± 0.2 band. Kept as running class sums: `update` is a rank-one update plus one solve (Woodbury below 384 training rows, Cholesky above), ~6 ms per add on the seed corpus, ~14 ms at 1,000. With fewer than 10 negatives it falls back to `TopicMargin`.
  - `TopicMargin` (fallback): contrastive margin of the substantive body against the generic-chatter centroid (`data/off_topic_anchors.json`). A margin ≤ 0 means off-topic.
- **Errors and logging**: raise the specific `novelty.errors` class (each is also a `ValueError` or `RuntimeError`). Library modules only call `logging.getLogger(__name__)`; handlers are configured by entry points through `logging_setup.configure_logging`. The scorer's per-request INFO lines are the contract the README documents, so keep them to one input line and one result line.
- **`scorer.py`**: orchestration. `submit()` adds a submission only if it's non-duplicate, substantive, has a relevance gate ≥ 0.5 and ≥ 20% of its substantive words in on-topic clauses (admitted comments train the gate). `add_many()` recalibrates once per batch. `fork()` deep-copies the state while sharing the embedder and preparer, which tests and the server's Reset rely on.
- **`evaluation.py`**: `python -m novelty eval` scores `data/eval/heldout.json` (200 items by independent authors, blind-checked labels) and reports pass rates per label, AUCs and the sequential check.
- **Invariant:** incremental `update()` must equal a full `fit()` on the same index. `test_incremental_calibration_matches_a_full_refit` guards this, so extend it when you add a stateful signal.

Embedding backends are in `novelty/embeddings.py`. `NOVELTY_EMBEDDER=local|gemini` selects one; the default is gemini if `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) is set, otherwise local bge-small, lazy-loaded. Tests force local through `tests/conftest.py`, and `NOVELTY_TEST_EMBEDDER=gemini` overrides that.

The web UI (`novelty/server.py`) logs every committed attempt to `data/user_submissions.json` (gitignored) and replays the admitted ones with `add_many` on startup. Never write them into `corpus.json`: the tests depend on that seed set. The server also listens on `::1`, because Windows resolves `localhost` to IPv6 first and a refused connect cost 200 ms per request. It refuses a non-loopback `Host` or foreign `Origin` (403) and non-JSON POSTs (415): that is the CSRF / DNS-rebinding protection, so the page must keep sending `Content-Type: application/json`, reset included.

## Data and test conventions

- **Evaluation discipline:** `data/eval/heldout.json` has a `dev` split (the only data any threshold or hyper-parameter may be tuned on) and a `test` split (reported in the README, never tuned on). `data/relevance_negatives.json` is training data for the relevance gate and must never be reused for evaluation; `test_heldout.py` checks the sets are disjoint.
- `data/probes.json` holds the labelled test inputs, grouped as `novel_relevant`, `duplicates`, `off_topic` and `off_topic_adjacent`. The behaviour tests index into these groups by position, so add new probes at the end of a group. Novel and off-topic probes must not appear in `corpus.json` or `off_topic_anchors.json` (`test_shape.py` enforces this).
- Test thresholds: novel ≥ 0.6, non-novel ≤ 0.25, off-topic ≤ 0.01. Some tests refer to corpus ids directly (`c01`, `c04`, `c20`, `c27`, and the crowded/one-off lists), and the `d_copy` / `d_stance_flip` probes are copies of `c01` / `c11`, so editing `corpus.json` can break them.
- `test_adversarial.py` and `test_edge_cases.py` pin behaviours that were once broken; where a case records a pre-fix score (a `# was:` comment or a docstring note), that is what it scored before the fix. Don't loosen them to make a change pass.
- The `scorer` fixture is a fork of one session-scoped build. Use `base_scorer.fork` wherever a test needs a scorer factory.
- Tests use the seed corpus by default (the whole-corpus default failed 11 tests once five web-UI comments were admitted). Use `helpers.assert_novel` for "this should be rewarded" assertions: in `NOVELTY_TEST_CORPUS=full` mode it treats probes already submitted through the web UI correctly. Use the `corpus_size` fixture, never a literal 50. Thresholds and the toy embedder live in `tests/helpers.py`.
- Known weak spots: low-overlap paraphrases score as new (held-out paraphrase pass rate 53%, sequential 0%); the learned gate blocks mixed half-off-topic comments and would block c36 / c49 as new submissions (strict `xfail` tests record all three: when one starts passing, remove its marker); the per-clause relevance test costs new ideas whose new clause is only loosely on-topic (test new-idea rate 48%); a quote of an existing comment dominates whole-text similarity; a mostly-foreign sentence earns nothing. See the README "Held-out evaluation" and limitations before "fixing" thresholds, and re-measure on dev.
- Spelling correction must stay deterministic: `candidates()` is a set, so anything chosen from it has to be sorted first (`test_spelling_correction_is_identical_across_processes`).

## Workflow

Commit after each feature or change and push to the user's GitHub remote. That's a standing instruction; `.venv/` and `.cache/` are gitignored. Measure before optimising: the README's Performance table records what was measured and what was deliberately left alone.
