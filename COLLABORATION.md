# How I worked with the coding agent

I built this with **Claude Code** (Claude Opus 5.5). My role was to set the requirements and push on quality: I had the agent attack its own work, stress-test the inputs, measure performance, and justify each design change with numbers. The agent proposed the technical design, implemented it, and ran the experiments behind each decision. The commit history follows the same sequence.

## The prompts that shaped the system

| # | What I asked for | What it produced, with the evidence |
|---|---|---|
| 1 | Build the pipeline described in the problem statement | A comment shape (headline, body, stance) responding to a 95-word article. Score = novelty × relevance gate, both calibrated against the corpus rather than fixed thresholds. Local embeddings with Gemini as an option. A clustered 50-comment synthetic corpus. Tests for the four required behaviours. |
| 2 | Walk me through the architecture; I summarised the pipeline in my own words first | Two corrections to my summary: novelty is normalised relative to the corpus's own spread (not min-max), and relevance is a separate gate that off-topic text must pass. |
| 3 | Act as a senior engineer trying to break the scorer, and keep the codebase modular | ~20 attacks were run before any fix, and six broke it: keyword stuffing (0.61), a "kitchen sink" of existing takes (0.81), word repetition (0.77), spam padding (0.66), restating the article (0.45–0.58), and disguised copies (undetected). Each was fixed and kept as a regression test. The scoring was restructured into independent, pluggable signal classes. |
| 4 | Stress-test the input shape: half English and half another language, under 10 and over 100 words, poor English. A relevant comment must not be penalised for its English. | Misspelled stock takes scored 0.86, because typos read as new vocabulary, so I got spelling correction and per-sentence language detection. Short novel ideas went from 0.05 to 0.83, generic praise from 0.41 to 0.01, and a stock take with an untranslated half from 0.49 to 0.11. Novel ideas in broken English stayed rewarded (0.55–0.89). |
| 5 | Profile performance and remove every unnecessary cost | Rewriting a JSON cache was 77% of request time, so the cache moved to SQLite (~90 ms → ~15 ms per new text). Recalibration went from O(n²) to incremental: 555 ms → 5.6 ms per insert at 1,000 comments, with a test proving it equals a full refit. A 200 ms-per-request Windows `localhost` delay was removed. The test suite went from 9.6 s to 2 s. |
| 6 | Add logging and error handling throughout | Typed errors with clear messages. Two log lines per input (the input, then the calculation with its timing). Request ids that tie errors to their log lines. Optional parts degrade instead of failing. It also caught a Windows bug that let two servers share one port. |
| 7 | Run the tests against the whole corpus, including submissions added through the web UI | The suite uses the seed plus admitted submissions by default. A probe that was already submitted is asserted as a copy, or skipped with the id of the submission covering it. |
| 8 | Clean up the code without removing any functionality | 60 changes, each checked by an independent reviewer. The scores, signals and reasons for 29 test inputs were byte-identical before and after. The review also surfaced two real bugs: the package build left out the signals module, and a caller's embedder was silently replaced. |
| 9 | A clean, professional, responsive frontend with two examples per stance | The essential result up front, with the full breakdown in a collapsed section. The examples are verified against the scorer, and a test keeps them honest. Choosing the examples exposed a relevance gap: an off-topic hiking comment scores 0.21 instead of 0. |

## How I checked the work

- **Numbers, not claims.** I asked for before-and-after measurements for every change, which are the figures above and in the README.
- **Adversarial testing of the agent's own tests.** Because the agent wrote both the corpus and the tests, I had it red-team the scorer and add every failure as a regression test.
- **Behaviour locked during refactoring.** For the cleanup, I required identical output on a fixed set of inputs, not just passing tests.

## What the agent generated

- **The data:** the 50 seed comments, the labelled test inputs, the off-topic calibration comments and the UI examples. No Gemini key was available during development. [scripts/generate_corpus.py](scripts/generate_corpus.py) regenerates a corpus of the same clustered shape with Gemini.
- **The code, tests and documentation,** following the direction above.

## Open items

- The thresholds (relevance gate 0.1 / 0.5, 0.6 hybrid weight, 6-token lexical threshold, 4-word specificity minimum) were tuned on this corpus and haven't been verified on Gemini embeddings.
- Nature and outdoor text near the "park" topic can partly pass the relevance gate: the hiking comment above scored 0.21.
- The pipeline is English-only; see the README's limitations section.
