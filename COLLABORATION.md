# How the coding agent was used

The brief allows a coding agent but asks for an explanation of how it was prompted. This solution was built with **Claude Code** (Claude Opus 5.5) in VS Code. This file records the prompts, the decisions the agent made on its own, and where evidence changed the design. The commit history (`git log`) follows the same sequence.

> **Author's note:** this is a draft reconstructed from the session. Add your own reasoning, review steps and any follow-up prompts before submitting.

## Prompts given

1. `/init` in an empty folder. The agent reported there was nothing to analyse and didn't invent a CLAUDE.md.
2. The full hackathon problem statement pasted with no extra instruction. The agent treated this as "build it" and picked defaults rather than asking questions, stating each one:
   - **Python**, the first preferred language in the brief.
   - **Local embeddings by default, Gemini when a key is present.** No `GEMINI_API_KEY` was available and tests have to be reproducible offline.
   - **Content shape:** a comment on a ≤100-word news brief, with `headline` + `body` + a multi-choice `stance`.
   - **Scoring:** `novelty × relevance_gate`, with both parts calibrated against the data rather than fixed thresholds.
3. *"also use git init and push the code to my github whenever a new feature or change is being made."* The agent initialised the repository and committed after each feature. It couldn't create the GitHub remote itself because the `gh` CLI wasn't installed, so it asked for a repository URL, and has pushed after every change since.
4. *"create a minimal frontend to test the project, keep it simple."* The agent built a standard-library HTTP server and a single static page, with no new dependencies.
5. *"check if the score and add to corpus is working, i dont see the new entries being added to the corpus.json file."* The add worked, but only in memory. The agent added persistence to a *separate* file (`data/user_submissions.json`) and explained why: the tests and the README's numbers are calibrated on the fixed `corpus.json`.
6. The author summarised the architecture in their own words and asked for a walkthrough. The agent confirmed most of it and corrected two points: normalisation is relative to the corpus rather than min-max, and the relevance gate was missing from the summary.
7. *"imagine you are a senior AI engineer who wants to break this code … keep the entire code base modular."* This led to the adversarial pass and the signal-based architecture (see below).
8. *"test the case where half the texts are in english and half in a different language … if the user input text is only a few words (<10) or … >100 … bad english … the reward must not be too low as the context is still relevant."* This led to the input edge-case pass.
9. *"focus on performance … an optimal performance project where no unnecessary time or effort is used."* This led to the measured performance pass.

## What the agent did, and where evidence changed the plan

The agent ran experiments before each design change, and the scratch results drove the decisions.

### Building the scorer

| Step | Observation | Change |
|---|---|---|
| First end-to-end demo | Copies, paraphrases and clearly off-topic probes behaved correctly, but a **same-town, different-subject** comment (Riverton High football) scored **0.90** | Relevance needed rethinking |
| Compared 4 local embedding models for raw similarity to the article | In every model the football comment fell *inside* the range of genuine on-topic comments | Raw similarity dropped as the relevance signal |
| Tried a topic keyword list built from the article and corpus | The football comment shares "Riverton" and "state" with the article, and giving article words extra weight made it *worse* | Dropped the keyword list |
| Tried a **contrastive margin** against generic off-topic comments | Football margin −0.07, below all 50 corpus comments (min +0.045) | Adopted, with `margin ≤ 0` as the boundary |
| "Novelty updates over time" test failed | A reworded repeat of an already-submitted idea still scored **0.60** | Investigated |
| Tried more weight on the nearest neighbour, then a bigger model (bge-base) | Best was 0.36; bge-base alone was 0.81 | Neither was enough |
| Tried **hybrid similarity** (embedding + TF-IDF) | Reworded repeat → 0.10, new ideas still > 0.8 | Adopted (60/40 blend) |

### Adversarial pass (prompt 7)

The agent wrote ~20 attacks and ran them against the pipeline *before* changing anything. It fixed only the ones that actually broke, and kept each as a regression test.

| Attack | Score before | Fix |
|---|---:|---|
| Keyword stuffing | 0.61 | relevance measured on substantive body clauses only |
| "Kitchen sink" restating every existing take | 0.81 | clause-level coverage. "Closeness to the average comment" was tried first and failed (z = −0.09) |
| "park" × 40 | 0.77 | content-quality check (function-word share, repetition) |
| Stock take + spam padding | 0.66 | clause coverage + relevance on content |
| Echo / paraphrase of the article | 0.45 / 0.58 | the article became a reference item |
| Homoglyph and zero-width-character copies | undetected | Unicode normalisation before validation |

The same pass restructured the code into independent `Signal` classes with one interface, so new checks plug in without touching the pipeline.

### Input edge cases (prompt 8)

| Case | Score before | Finding → fix |
|---|---:|---|
| Stock take with heavy typos | **0.86** | Misspelled words are unseen tokens, so they read as novelty → spelling correction (with a dictionary-based language check first; that was replaced after it called a typo-heavy headline "foreign" and "corrected" Spanish into "La call Elm se inundate") |
| English stock take + Spanish novel idea | 0.49 | Untranslated text made the embedding look unusual → similarity uses English content only |
| Short novel idea ("Put EV chargers at the outer shuttle lot.") | 0.05 | Clause coverage is unreliable for a lone clause. Four alternatives were measured and none separated the cases → it now applies only to multi-clause text |
| Short generic praise | 0.41 | → a specificity modifier |
| Short stock take with rare words | 0.32 | → TF-IDF weight shrinks for very short text. Swept 4–10 tokens; 6 separated short novel (≥ 0.83) from short stock (≤ 0.08) |
| Novel ideas in broken English | 0.55–0.89 | Already fine; now pinned by tests so they stay that way |

### Performance (prompt 9)

The agent profiled first: startup stages, per-request latency on new versus cached text, a cProfile of `submit()`, and scaling from 50 to 1,000 entries with a fast fake embedder. The measured bottlenecks, in order:

1. The JSON embedding cache was rewritten whole on every new text: 77% of request time, growing without bound. → SQLite.
2. Recalibration was O(n²) per insert (555 ms at 1,000 entries), mostly re-tokenising every stored clause. → incremental updates with an inverted index and amortised refits, plus a test that incremental equals a full refit.
3. A flat +200 ms on every HTTP request, traced to Windows resolving `localhost` to IPv6 first. → also listen on `::1`.
4. The test suite spent 8 of its 9.6 s in `httpd.shutdown()` poll waits. → short poll interval, and forking one pre-built scorer.

ONNX thread count and texts-per-request were measured too, and deliberately left alone because there was no gain. The README's Performance table has the before and after numbers.

## Agent-generated content

- **The 50-item corpus, probes and off-topic anchors** were written by the agent (itself an LLM), since no Gemini key was available during development. The brief recommends LLM-generated synthetic content. The agent was deliberate about the corpus's shape: clustered, with many repeats of the obvious takes. [scripts/generate_corpus.py](scripts/generate_corpus.py) regenerates a corpus of the same shape with Gemini.
- **The probes and attack cases** were written to be disjoint from the corpus and the calibration anchors, and a test enforces this for the new-idea and off-topic probes.

## What a human should still review

- The **thresholds**: the relevance gate (0.1 / 0.5), the 0.6 hybrid weight, the 6-token lexical threshold and the 4-word specificity minimum. They were chosen from the experiments above on *this* corpus. Most are expressed relative to the data, so they should carry over to other corpora and to Gemini, but that hasn't been verified with a live key.
- The limitations listed at the end of the [README](README.md): English only, short question-style comments get only partial relevance, and quality isn't judged beyond "makes a statement".
