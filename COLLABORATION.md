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
3. Mid-task: *"also use git init and push the code to my github whenever a new feature or change is being made."* The agent initialised the repository and committed after each feature. It couldn't create the GitHub remote itself because the `gh` CLI wasn't installed, so it asked for a repository URL.

## What the agent did, and where evidence changed the plan

The agent ran experiments before each design change, and the scratch results drove the decisions:

| Step | Observation | Change |
|---|---|---|
| First end-to-end demo | Copies, paraphrases and clearly off-topic probes behaved correctly, but a **same-town, different-subject** comment (Riverton High football) scored **0.90** | Relevance needed rethinking |
| Compared 4 local embedding models (bge-small, bge-base, nomic, MiniLM) for raw similarity to the article | In every model the football comment fell *inside* the range of genuine on-topic comments | Raw similarity dropped as the relevance signal |
| Tried a topic keyword list built from the article and corpus | Separated some cases, but the football comment shares "Riverton" and "state" with the article, and giving article words extra weight made it *worse* | Dropped the keyword list |
| Tried a **contrastive margin**: similarity to the topic centroid minus similarity to the centroid of generic off-topic comments | Football margin −0.07, below all 50 corpus comments (min +0.045) | Adopted, with `margin ≤ 0` as the boundary |
| Wrote the behavioural tests. The "novelty updates over time" test failed | A reworded version of an already-submitted idea still scored **0.60**: the small model rates it barely closer than a genuinely new idea | Investigated |
| Tried giving more weight to the nearest neighbour, and a bigger model (bge-base) | Best was 0.36; bge-base alone was 0.81 | Neither was enough |
| Tried **hybrid similarity** (embedding + TF-IDF) | Reworded repeat → 0.10, new ideas still > 0.8, worst paraphrase 0.07 | Adopted (60/40 blend) |

## Agent-generated content

- **The 50-item corpus, probes and off-topic anchors** were written by the agent (itself an LLM), since no Gemini key was available during development. The brief recommends LLM-generated synthetic content. The agent was deliberate about the corpus's shape: clustered, with many repeats of the obvious takes. [scripts/generate_corpus.py](scripts/generate_corpus.py) regenerates a corpus of the same shape with Gemini.
- **The probes** (new ideas, copies and paraphrases, off-topic) were written to be disjoint from the corpus and the calibration anchors, and a test enforces this for the new-idea and off-topic probes.

## What a human should still review

- The **gate thresholds** (`relevance_floor=0.1`, `relevance_full=0.5`) and the 0.6 hybrid weight were chosen from the experiments above on *this* corpus. They are expressed relative to the data, so they should carry over to other corpora and to Gemini, but that hasn't been verified with a live key.
- The limitations listed at the end of the [README](README.md): short question-style comments get only partial relevance, keyword stuffing is possible, and quality isn't scored.
