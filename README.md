# Rewarding novelty in user generated content

A pipeline that scores a new user submission on a normalised **[0.0, 1.0]** reward. It rewards submissions that say something the existing ~50 submissions haven't said, as long as they're still a response to the article. Off-topic content earns nothing, however original it is.

```
score = novelty × relevance_gate
```

Everything runs offline on a local embedding model (no API key needed). If `GEMINI_API_KEY` is set, it switches to Gemini embeddings automatically; that path hasn't been tested with a live key.

## At a glance

- **Content shape:** a reader comment (headline, body, and a four-way stance) on a 95-word local-news article, compared against a clustered corpus of 50 comments.
- **Novelty** is measured *relative to the corpus*: a submission's distance to its nearest existing comments is compared with how far a typical corpus comment is from the rest. So 0.5 means "as novel as a typical comment", and there are no hard-coded cosine thresholds.
- **Relevance** is learned for this article: a regularised discriminant separates on-topic comments from 72 same-town *off-topic* ones (bus routes, the library, water rates…) that an independent author wrote.
- **Held-out evaluation** on 130 labelled items the method was never tuned on:
  - **91% of off-topic comments are blocked** (up from 33% with the previous gate), including 100% on topics the training data never covered;
  - 55% of new ideas and 50% of paraphrases are handled correctly;
  - see [Held-out evaluation](#held-out-evaluation) for the full picture, including what still fails.
- **Deliverables:** 189 automated tests, a CLI, a JSON API and a responsive web UI.
- **Plain-language design notes:** [docs/DESIGN.md](docs/DESIGN.md) ([PDF](docs/DESIGN.pdf)) cover the design, rationale, success criteria, results and limitations without the jargon.

## 1. The content shape

**Fixed content** ([data/fixed_content.json](data/fixed_content.json), 95 words): Riverton City Council votes to demolish a 600-space downtown parking garage and build a $14M park, funded by a business levy and state grants, with a construction shuttle and a public comment window.

**User submission** ([novelty/models.py](novelty/models.py)): three discrete, user-provided properties.

| Property   | Type                                                 | Validation     |
|------------|------------------------------------------------------|----------------|
| `headline` | free text                                            | 1–120 chars    |
| `body`     | free text                                            | 20–2000 chars  |
| `stance`   | multi-choice: `support` · `oppose` · `mixed` · `undecided` | enum     |

Text is normalised before validation: Unicode NFKC, invisible characters removed, and look-alike Cyrillic/Greek letters mapped to Latin in otherwise Latin-script text. That stops disguised copies and zero-width padding.

A **text preparation** step ([novelty/preparation.py](novelty/preparation.py)) then handles each sentence:
- **Language:** it detects whether the sentence is English from function words and script, and excludes non-English sentences from scoring.
- **Slang:** it expands texting shorthand.
- **Spelling:** it corrects single-edit typos, preferring words from the article and corpus. Corrections are deterministic across runs.

Only the analysis sees the prepared text; the display text never changes.

**Corpus** ([data/corpus.json](data/corpus.json)): 50 synthetic comments written by an LLM, deliberately clustered the way real comment sections are:
- 10 variations of "losing parking will kill Main Street";
- 9 of "great for families";
- 7 about cost, 6 about heat and trees, 5 about the shuttle, 5 about safety and upkeep, 4 about foot traffic;
- and a handful of one-offs.

A corpus where every comment is unique would make "novel" meaningless. [scripts/generate_corpus.py](scripts/generate_corpus.py) regenerates a corpus of this shape with Gemini.

## 2. How reward is decided

The score is built from independent **signals**, each in [0, 1]:

```
novelty = min(novelty signals) × Π(modifier signals)
score   = novelty × Π(relevance signals)
```

| Signal | Kind | What it measures |
|---|---|---|
| `whole_text` | novelty | Hybrid-similarity distance of the whole submission to its nearest neighbours, relative to the corpus |
| `clause_coverage` | novelty | Novelty of the most novel *relevant, substantive* clause (applies to multi-clause text) |
| `duplicate` | modifier | Near-copy of a submission **or of the article itself** → 0 |
| `quality` | modifier | Keyword lists, word repetition, no sentence-like content, unsupported language → 0 |
| `specificity` | modifier | Fewer than 4 specific content words (generic praise) → scaled down |
| `stance` | modifier | Stance rarity: ×0.9 for the most common stance, up to ×1.0 for the rarest |
| `relevance` | relevance | Learned, article-specific relevance of the substantive body → smooth gate |

Taking the **min** of the novelty signals means both views must agree that something is new. **Multiplying** by relevance means high novelty can't make up for being off-topic, and being on-topic can't make up for being a repeat.

### Novelty

1. **Hybrid similarity** to every reference item (the corpus plus the article) = `0.6 × embedding cosine + 0.4 × TF-IDF cosine`. The TF-IDF share shrinks for very short text.
2. **Raw novelty** = `0.5 × (1 − nearest-neighbour similarity) + 0.5 × (1 − mean top-5 similarity)`: "has someone already said this?" plus "is this a crowded topic?"
3. **Calibrated against the corpus itself.** Every corpus comment's raw novelty is computed leave-one-out. A new submission's value is then converted to a robust z-score (median/MAD) and passed through the normal CDF. This works with any embedding model.
4. **Clause coverage** scores each relevant, substantive clause the same way. That catches a "kitchen sink" comment restating many existing takes, and a stock take padded with unrelated text.
5. **Modifiers:**
   - A copy (≥ 60% character 5-gram containment) gets 0, and so does text that makes no statement.
   - Stance multiplies novelty by 0.9–1.0, so it can never rescue a copy.

### Relevance

The first two approaches failed on this problem:
- **Raw embedding similarity to the article** measures "reads like a local comment", not "responds to this article".
- **The previous gate** measured whether a comment was closer to the topic than to generic lifestyle chatter. An independent evaluation panel showed that it let **same-town civic comments** through: "Route 7 bus cuts" and "Library hours" both scored 0.91. In this embedding space every Riverton civic comment sits in one narrow similarity band. Off-the-shelf NLI and reranker models were also tried, and both ranked a clearly on-topic parking comment below off-topic ones.

The current gate ([`TopicDiscriminant`](novelty/signals/relevance.py)) **learns what this article's topic is** from labelled examples:
- **Positives:** the article and every on-topic comment (the corpus, plus admitted submissions).
- **Negatives:** 72 same-town comments about *other* subjects ([data/relevance_negatives.json](data/relevance_negatives.json)), plus the 10 generic anchors. The negatives cover transit, library and schools, utilities, other council business, other parks and garages, unrelated businesses and events, roads, and nature. An independent author wrote them, and a blind annotator confirmed each one is off-topic.
- **The model:** a shrinkage Fisher discriminant, one direction in embedding space that best separates the two sets. The covariance is regularised toward the identity (shrinkage 0.9), and the boundary is placed where it best separates the training data; the gate rises across a narrow band around it. Shrinkage and band width were chosen on the evaluation set's **dev** split, and were stable when the negatives were resampled.
- **Cost of a refit:** it's computed in Woodbury form (a ~130 × 130 solve instead of 384 × 384), so each admitted submission costs ~4 ms.

Relevance is judged on the substantive body only, so stuffing the headline or appending keywords can't buy it. Without at least 10 negatives, the signal falls back to the previous contrastive margin (`TopicMargin`).

### What joins the corpus

`submit()` only admits relevant, non-duplicate, substantive submissions. Admitted ones become reference data and new positives for the relevance gate, so floods of spam or copies can't shift the calibration.

## 3. Held-out evaluation

`python -m novelty eval` runs the scorer over [data/eval/heldout.json](data/eval/heldout.json): **200 labelled items**.

**Where they came from:**
- Independent authors wrote them, seeing only the article and the seed corpus, never the scorer or its tests. They include:
  - new ideas, some inside crowded clusters and some short, long or in poor English;
  - low-overlap paraphrases of specific comments;
  - generic comments;
  - off-topic comments on seen topics, on topics absent from the training negatives, on distant subjects, and "tricky" ones mentioning parking in passing;
  - idea/rewording pairs.
- The panel's 60 red-team items are included too.
- A blind second annotator re-labelled every item without seeing the original label, and they agreed on all 200.

**How it was used:** the items are split into **dev** (70), the only data any setting was tuned on, and **test** (130), which no setting was tuned on. All 60 red-team items are in test, and they include the failures that prompted the learned gate (bus cuts, library hours), so the off-topic figures are somewhat flattered. The table below is test.

| Test split (130 items) | Previous gate | **Learned gate** |
|---|---:|---:|
| Off-topic blocked (score ≤ 0.01) | 33% | **91%** |
| · same-town topics the negatives covered | 0% | **89%** |
| · same-town topics the negatives never covered | 50% | **100%** |
| · distant topics | 100% | **100%** |
| Tricky items (own expectations) | 71% | **86%** |
| Relevance AUC (on-topic vs off-topic) | 0.79 | **0.91** |
| New ideas rewarded (≥ 0.6) | 66% | 55% |
| Paraphrases not rewarded (≤ 0.25) | 53% | 50% |
| Generic comments not rewarded | 80% | 80% |
| AUC new vs not new | 0.79 | 0.78 |
| Sequential: a rewording is new until the idea is submitted, then not | 0% | 0% |

**What still fails, in the order it matters:**
- **Low-overlap paraphrases and rewordings.** A comment that restates an existing point in mostly different words often scores as new. That's why only half of the paraphrases are caught, and why the sequential check fails. It passes when the rewording shares distinctive words ("stormwater sponge": 0.86 → 0.06 once the idea is submitted). The fix would be a "same point?" check on the top neighbours, such as a cross-encoder or an LLM.
- **Off-topic leaks that remain** are the topic's nearest neighbours: other parks (a dog-run fence, cracked tennis courts), library funding, and recycling pickup.
- **The price of the stricter gate:**
  - New-idea reward fell 11 points: a few genuine ideas that sit near an off-topic subject, such as falcons nesting on the garage, are now gated out.
  - Comments that are *half* unrelated chatter now score 0; the half-and-half case scored 0.74 before.
  - Two seed comments would be blocked if they arrived fresh: c36, a short shuttle question, and c49, a process complaint about the vote.
- **Label caveat:** the authors and annotators are the same model family as the one that built the scorer. The labels are machine-verified, not human-verified.

`tests/test_heldout.py` pins this behaviour:
- at least 85% of off-topic items blocked and relevance AUC at least 0.85 on test, plus the improvement over the previous gate;
- floors for new ideas and paraphrases set about 10 points below the measured values, as regression guards rather than quality claims.

## 4. Probe results and adversarial testing

`python -m novelty demo` on the labelled probes (local model):

| Probe | What it is | Whole-text novelty | Relevance | **Score** |
|---|---|---:|---:|---:|
| `n_flood` | design the park as stormwater retention | 0.94 | 1.00 | **0.84** |
| `n_carbon` | keep the garage frame, terraced park on top | 0.90 | 1.00 | **0.85** |
| `n_depot` | the site was the 1880s rail depot: add a heritage walk | 0.95 | 1.00 | **0.85** |
| `d_copy` | copy of c01 with light edits | 0.00 | 1.00 | **0.00** |
| `d_paraphrase_parking` | reworded "no parking = no customers" | 0.01 | 1.00 | **0.01** |
| `d_stance_flip` | exact copy of c11 with the rarest stance | 0.00 | 1.00 | **0.00** |
| `o_football` | Riverton High football (same town, different subject) | 1.00 | 0.00 | **0.00** |
| `oa_bus` … `oa_school` | six same-town civic comments (bus route, library hours, water rates, snow plowing, polling place, school start times) | 0.99–1.00 | 0.00 | **0.00** (0.45–0.97 with the previous gate) |

Attacks that once broke the pipeline, each now a regression test in [tests/test_adversarial.py](tests/test_adversarial.py):

| Attack | Before | Now | Defence |
|---|---:|---:|---|
| Keyword stuffing ("…sourdough… garage park Elm Street levy") | 0.61 | **0.00** | relevance measured on the substantive body only |
| "Kitchen sink" listing every existing take | 0.81 | **0.23** | clause coverage |
| "park" × 40 | 0.77 | **0.00** | content-quality check |
| Stock take padded with crypto spam | 0.66 | **0.00** | clause coverage + relevance on content |
| Echoing / paraphrasing the article | 0.45 / 0.58 | **0.00** | the article is a reference item |
| Copies using look-alike letters or zero-width characters | undetected | **caught** | text normalisation |
| Stock take with heavy typos ("Withot the garaje peple cant park…") | 0.86 | **0.02** | spelling correction |

Input edge cases ([tests/test_edge_cases.py](tests/test_edge_cases.py)):

| Case | Score |
|---|---:|
| English new idea + Spanish / French / Hindi text | 0.81 (foreign text excluded) |
| English stock take + Spanish new idea | 0.11 (the untranslated half can't supply novelty) |
| Short new idea: "Put EV chargers at the outer shuttle lot." | 0.83 |
| Short stock take / generic praise | 0.08 / 0.01 |
| New idea in broken English, heavy typos, texting style | 0.55–0.89 |
| Half off-topic chatter + half new idea | 0.00 (trade-off of the learned gate, see above) |

## 5. Automated tests

```
pytest            # 189 tests, a few seconds after the first model download
```

| Requirement from the brief | Where it is tested |
|---|---|
| Truly novel content is rewarded | `test_behavior.py::test_truly_novel_relevant_content_is_rewarded`; held-out floors in `test_heldout.py` |
| Non-novel content is not rewarded | `test_non_novel_content_is_not_rewarded`, copy/stance-flip/padding tests, `test_adversarial.py` |
| **High novelty, low relevance is not rewarded** | `test_high_novelty_but_low_relevance_is_not_rewarded` (novelty ≥ 0.8 **and** score ≤ 0.01); `test_same_town_civic_comments_about_other_subjects_are_not_rewarded`; `test_heldout.py::test_off_topic_content_is_not_rewarded_on_unseen_inputs` |
| Remain relevant to the fixed content | `test_relevance_gate_keeps_nearly_all_genuine_responses` (≥ 90% of corpus comments keep a full gate, leave-one-out) |
| Novelty is relative to other submissions | `test_novelty_is_relative_to_what_has_been_submitted`, `test_crowded_takes_are_less_novel_than_one_off_takes_within_the_corpus` |
| Normalised to [0, 1] | `test_all_scores_are_normalised`, model-free invariants in `test_math.py` |

**The other test files:**
- `test_heldout.py`: evaluation-set integrity and aggregate bars.
- `test_edge_cases.py`, including a check that spelling correction is identical across processes.
- `test_math.py`: the maths, calibration guards, and that incremental equals a full refit. It uses a toy embedder, so it needs no model.
- `test_errors_logging.py`, `test_server.py`, `test_shape.py` and `test_examples.py`.

**Test corpus:** by default the tests run against the whole corpus, meaning the seed comments plus admitted web-UI submissions from `data/user_submissions.json`. A probe that was already submitted is asserted as a copy, or skipped with the covering id. `NOVELTY_TEST_CORPUS=seed pytest` pins the seed corpus for reproducible runs. Tests use the local model; `NOVELTY_TEST_EMBEDDER=gemini` would run them on Gemini.

## 6. Running it

```bash
python -m venv .venv
.venv/Scripts/activate                     # Windows; `source .venv/bin/activate` elsewhere
pip install -e ".[dev,gemini]"

pytest
python -m novelty serve                    # web UI at http://127.0.0.1:8000
python -m novelty eval --split test        # held-out evaluation (dev | test | all, --json)
python -m novelty demo                     # score all labelled probes
python -m novelty corpus                   # leave-one-out novelty of each corpus item
python -m novelty score --stance support \
  --headline "Put solar canopies over the shuttle lot" \
  --body "The outer shuttle lot is acres of bare asphalt; solar canopies would shade cars and help pay for the park."

# Optional: Gemini (free key at https://aistudio.google.com/); untested with a live key
export GEMINI_API_KEY=...
python scripts/generate_corpus.py          # regenerate a synthetic corpus → data/corpus.generated.json
```

**The web UI** (standard library only, responsive, light and dark themes):
- **What it shows:** the article, the three-field form, two examples per stance ([data/examples.json](data/examples.json), verified by `tests/test_examples.py`), and the result. The result gives the score, a plain-language verdict, novelty and relevance bars, and the most similar existing comment. A collapsed **Details** section holds the formula, every signal and the nearest comments.
- **Buttons:**
  - **Score** leaves the corpus unchanged.
  - **Score and add to corpus** applies the admission policy and logs the attempt to `data/user_submissions.json` (gitignored). Admitted submissions are replayed on restart.
  - **Reset my submissions** clears them.

**Environment variables:**

| Variable | Effect |
|---|---|
| `NOVELTY_EMBEDDER=local\|gemini` | Chooses the embedding backend. |
| `NOVELTY_CACHE_DIR` | Sets the model and embedding cache location (default `.cache/`). |
| `NOVELTY_LOG_LEVEL` / `NOVELTY_LOG_FILE` | Logging level, and an optional log file. |
| `NOVELTY_TEST_EMBEDDER` / `NOVELTY_TEST_CORPUS` | Test backend and test corpus. |
| `GEMINI_MODEL` | Model for the corpus generator. |

The first run downloads `BAAI/bge-small-en-v1.5` (~70 MB ONNX) into `.cache/`.

## 7. Logging and errors

Every scored input leaves two INFO lines: the input, and the result with its calculation and timing. `-v` (DEBUG) adds every signal, the nearest neighbours, clause counts and spelling corrections:

```
INFO    [-] novelty.scorer: submit input: id=u03 stance=support words=19 headline="Design the park to soak up floods"
INFO    [-] novelty.scorer: submit result: id=u03 score=0.823 = novelty 0.823 x gate 1.00 | whole 0.91 clause 0.95 | relevance 1.00 margin +41.873 | added to corpus | 8 ms (embed 0 ms, 0 new)
```

These lines were captured from a script, so they carry no request id (`[-]`); in the web server each line carries the request's id, like `[r000007]`. The margin is the submission's distance from the learned relevance boundary, in discriminant units: negative means off-topic.

**Errors:** every error is a `NoveltyError` ([novelty/errors.py](novelty/errors.py)).
- `ValidationError` for bad input → HTTP 400.
- `DataError` names the file and item.
- `EmbeddingError` for a model or network failure → HTTP 503. Gemini retries transient errors only.
- `CalibrationError` when the reference data can't support a calibration.
- `ScoringError` names the failing signal.

The request id ties the log lines of one web request together and is returned in every error response. Optional parts (the embedding cache, spelling correction, saving the submissions log) degrade with a warning instead of failing the request.

## 8. Architecture

```
novelty/
  models.py        Submission / FixedContent / ScoreBreakdown (validation + normalisation)
  text.py          normalisation, clause splitting, substantive-clause check, shingles
  preparation.py   per-sentence language detection, slang expansion, deterministic spelling correction
  embeddings.py    local (fastembed, lazy-loaded) and Gemini backends, SQLite vector cache
  lexical.py       sparse TF-IDF with an inverted index and amortised refits
  index.py         ReferenceIndex: article, admitted submissions, anchors and relevance negatives
  signals/         one class per signal: fit / update / evaluate
    novelty.py     WholeTextNovelty, ClauseCoverage
    modifiers.py   DuplicateCheck, ContentQuality, Specificity, StanceRarity
    relevance.py   TopicDiscriminant (default), TopicMargin (fallback)
  scorer.py        NoveltyScorer: analyse → evaluate signals → combine → admission policy
  evaluation.py    held-out evaluation: pass rates, AUCs, sequential check
  data.py          loaders for data/*.json, build_scorer()
  errors.py, logging_setup.py, __main__.py (CLI), server.py + static/index.html
data/
  corpus.json, fixed_content.json, off_topic_anchors.json, relevance_negatives.json,
  probes.json, examples.json, eval/heldout.json
```

**Adding a component** (an LLM judge, a toxicity filter, a language check) means subclassing `Signal`, choosing a `Kind`, and passing it in with `NoveltyScorer(signals=[*default_signals(config), MySignal()])`. A signal that needs calibration implements `fit(index)` and, optionally, an incremental `update(index, added)`.

## 9. Performance

Every optimisation was measured first.

| What | Before | After | Change |
|---|---:|---:|---|
| `score()` of new text (local model) | ~90 ms | **~15–20 ms** | SQLite embedding cache instead of rewriting a JSON file on every miss |
| HTTP via `localhost` on Windows | +200 ms | **+0.5 ms** | also listen on IPv6 loopback |
| add + recalibrate, 1,000-entry corpus | 555 ms | **5.6 ms** | incremental calibration (tested equal to a full refit) |
| add, including the relevance refit | 15 ms | **~4 ms** | Woodbury form of the discriminant (exact to 1e-15) |
| Server start, warm cache | ~1.3 s | **0.3 s** | lazy model load, batch replay |
| Test suite | 9.6 s | **~4–7 s** | short server poll interval, forked scorer fixture |

## 10. Known limitations

- **Paraphrases in different words** and the sequential "new until someone says it" property are the main quality gap (see [Held-out evaluation](#held-out-evaluation)).
- **The closest neighbours of the topic still leak:** other parks, library funding and recycling. Mixed comments that are half off-topic now earn nothing, and short questions and process complaints (c36, c49) are at risk of being gated out.
- **English only.** Non-English sentences are excluded, so they're neither rewarded nor penalised. Code-mixed sentences earn nothing.
- **Machine-labelled evaluation:** no human labels yet, and there's one article only. The relevance negatives are specific to this article, so a new article needs its own; with fewer than 10, the gate falls back to the contrastive margin.
- **Gemini is untested** with a live key.
- **Score oracle and moderation.** `/api/score` can be used to iterate on wording, and abusive or false comments are judged only on novelty. A deployment would add rate limiting and a moderation signal.
- **Scale.** Each insert is O(n). At tens of thousands of entries, add an approximate-nearest-neighbour index behind `ReferenceIndex`.
