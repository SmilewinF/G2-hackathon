# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Hackathon solution (G2, "Rewarding novelty in submissions"): score a user submission's novelty against ~50 others on [0, 1], while giving zero reward to submissions that are novel only because they're off-topic. The README is the submission write-up. COLLABORATION.md documents how the agent was prompted, as the brief requires, so update it when a design decision changes.

## Commands

```bash
.venv/Scripts/python -m pip install -e ".[dev,gemini]"   # venv lives in .venv (Windows layout)
.venv/Scripts/python -m pytest                            # full suite
.venv/Scripts/python -m pytest tests/test_behavior.py::test_same_town_different_subject_is_not_rewarded
.venv/Scripts/python -m novelty demo                      # score table for data/probes.json — check after any scoring change
.venv/Scripts/python -m novelty corpus                    # leave-one-out novelty per corpus item
.venv/Scripts/python -m novelty score --headline ... --body ... --stance support [--json]
```

The first run downloads the fastembed ONNX model into `.cache/fastembed`. Embeddings are cached in `.cache/embeddings/` by model and text hash. Delete that folder if you change how submission text is built.

## Architecture

`score = novelty × relevance_gate`, implemented in `novelty/scorer.py` (`NoveltyScorer`). The key idea is that **nothing is a fixed cosine threshold**. Every raw signal is normalised against a distribution recomputed in `_calibrate()` whenever the corpus changes:

- **Novelty**: hybrid similarity (`dense_weight` × embedding cosine + the rest × TF-IDF cosine from `lexical.TfidfIndex`). Raw novelty blends nearest-neighbour and top-k distance, is z-scored against the corpus's own *leave-one-out* median and MAD, then passed through the normal CDF. Character-shingle containment (`lexical.containment`) forces novelty to 0 for copies. Stance rarity is a small multiplier, never additive, so a stance flip can't rescue a copy.
- **Relevance**: contrastive margin `sim(topic centroid) − sim(generic centroid)`. The topic is the fixed content plus on-topic corpus members. The generic centroid is `data/off_topic_anchors.json`. A margin ≤ 0 means off-topic. The margin is divided by the median leave-one-out corpus margin, then passed through a smoothstep gate.
- `submit()` = score, then add to the corpus. Every submission affects future novelty, but only gated-in ones join the topic centroid (`_topic_member`), which keeps spam from moving the topic.

Embedding backends are in `novelty/embeddings.py`. `NOVELTY_EMBEDDER=local|gemini` selects one; the default is gemini if `GEMINI_API_KEY` is set, otherwise local bge-small. Tests force local through `tests/conftest.py`. `NOVELTY_TEST_EMBEDDER=gemini` overrides that.

## Data and test conventions

- `data/probes.json` holds the labelled test inputs, grouped as `novel_relevant`, `duplicates` and `off_topic`. The behaviour tests index into these groups by position, so add new probes at the end of a group. Novel and off-topic probes must not appear in `corpus.json` or `off_topic_anchors.json` (`test_shape.py` enforces this).
- `tests/test_behavior.py` thresholds: novel ≥ 0.6, non-novel ≤ 0.25, off-topic ≤ 0.01. Some tests refer to corpus ids directly (`c01`, `c11`, `c27`, and the crowded/one-off lists). Editing `corpus.json` can break them.
- Known weak spot: short question-style comments (`c36`) get only partial relevance. See the README limitations section before "fixing" the gate thresholds.

## Workflow

Commit after each feature or change and push to the user's GitHub remote. That's a standing instruction; `.venv/` and `.cache/` are gitignored.
