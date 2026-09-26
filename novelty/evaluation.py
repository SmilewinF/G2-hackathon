"""Held-out evaluation: how the scorer behaves on labelled inputs it was not tuned on.

    python -m novelty eval [--split dev|test|all] [--failures N] [--json]

The set lives in data/eval/heldout.json. Its items were written by independent authors who saw
only the article and the seed corpus (never the scorer or its tests), and every label was
confirmed by a blind second annotator. Items are split into "dev" (the only data any threshold
may be tuned on) and "test" (reported, never tuned on); the red-team items are all in "test".

Metrics
- pass rate per label, against the project's thresholds:
  novel / seq_first >= 0.6, paraphrase / generic <= 0.25, off_topic <= 0.01
  (off-topic is also reported at <= 0.05, and "tricky" items carry their own expectation)
- AUC novel-vs-not: how well the score ranks new ideas above paraphrases and generic comments
- AUC relevance: how well the relevance gate ranks on-topic items above off-topic ones
- sequential: an idea scores >= 0.6 until someone submits it, then its rewording scores <= 0.25
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

from .data import DATA_DIR
from .errors import DataError
from .models import Submission
from .scorer import NoveltyScorer

EVAL_FILE = DATA_DIR / "eval" / "heldout.json"
NOVEL_MIN, NOT_NOVEL_MAX, OFF_TOPIC_MAX, OFF_TOPIC_LOOSE = 0.6, 0.25, 0.01, 0.05
THRESHOLDS = {  # label -> (comparison, bound)
    "novel": (">=", NOVEL_MIN), "seq_first": (">=", NOVEL_MIN),
    "paraphrase": ("<=", NOT_NOVEL_MAX), "generic": ("<=", NOT_NOVEL_MAX),
    "off_topic": ("<=", OFF_TOPIC_MAX),
}
ON_TOPIC = ("novel", "seq_first", "paraphrase", "generic")


@dataclass(frozen=True)
class EvalItem:
    id: str
    label: str
    submission: Submission
    split: str
    topic: str = ""
    of: str = ""
    expect: tuple[str, float] | None = None  # tricky items: their own (comparison, bound)

    def passes(self, score: float) -> bool | None:
        rule = self.expect or THRESHOLDS.get(self.label)
        if rule is None:
            return None
        op, bound = rule
        return score >= bound if op == ">=" else score <= bound


@dataclass
class EvalReport:
    split: str
    counts: dict[str, int] = field(default_factory=dict)
    pass_rates: dict[str, float] = field(default_factory=dict)
    off_topic_loose: float | None = None  # share of off-topic items at <= 0.05
    auc_novelty: float | None = None
    auc_relevance: float | None = None
    sequential: float | None = None
    by_topic: dict[str, float] = field(default_factory=dict)  # off-topic pass rate per topic group
    failures: list[dict] = field(default_factory=list)


def _parse_rule(rule: str) -> tuple[str, float]:
    op, bound = rule.split()
    if op not in (">=", "<="):
        raise DataError(f"unsupported expectation {rule!r}")
    return op, float(bound)


def load_eval_set(path: Path = EVAL_FILE, split: str = "all") -> list[EvalItem]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise DataError(f"evaluation set not found: {path}") from None
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise DataError(f"{path.name} is not valid JSON: {e}") from None
    items = []
    for i, d in enumerate(raw.get("items", []) if isinstance(raw, dict) else []):
        try:
            sub = Submission(headline=d["headline"], body=d["body"], stance=d["stance"], id=None)
        except (KeyError, ValueError) as e:
            raise DataError(f"{path.name} item {i} ({d.get('id')!r}): {e}") from None
        if split != "all" and d.get("split") != split:
            continue
        items.append(EvalItem(
            id=d["id"], label=d["label"], submission=sub, split=d.get("split", ""),
            topic=d.get("topic", ""), of=d.get("of", ""),
            expect=_parse_rule(d["expect"]) if d.get("expect") else None,
        ))
    if not items:
        raise DataError(f"{path.name} has no items for split {split!r}")
    return items


def auc(positives: Sequence[float], negatives: Sequence[float]) -> float | None:
    """Probability that a random positive scores above a random negative (ties count half)."""
    if not positives or not negatives:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in positives for n in negatives)
    return wins / (len(positives) * len(negatives))


def evaluate(make_scorer: Callable[[], NoveltyScorer], items: Sequence[EvalItem], split: str = "all") -> EvalReport:
    """Score every item against a fresh scorer (nothing is admitted), then run the sequential
    pairs, each on its own fork so pairs cannot influence each other."""
    base = make_scorer()
    report = EvalReport(split=split)
    results: dict[str, tuple[EvalItem, float, float]] = {}
    for item in items:
        r = base.score(item.submission)
        results[item.id] = (item, r.score, r.relevance_gate)
        ok = item.passes(r.score)
        report.counts[item.label] = report.counts.get(item.label, 0) + 1
        if ok is False:
            report.failures.append({"id": item.id, "label": item.label, "topic": item.topic, "score": r.score,
                                    "relevance_gate": r.relevance_gate, "headline": item.submission.headline})

    def rate(label: str) -> float | None:
        judged = [it.passes(s) for it, s, _ in results.values() if it.label == label]
        judged = [j for j in judged if j is not None]
        return sum(judged) / len(judged) if judged else None

    for label in sorted(report.counts):
        value = rate(label)
        if value is not None:
            report.pass_rates[label] = round(value, 3)
    off = [(it, s) for it, s, _ in results.values() if it.label == "off_topic"]
    if off:
        report.off_topic_loose = round(sum(s <= OFF_TOPIC_LOOSE for _, s in off) / len(off), 3)
        groups: dict[str, list[bool]] = {}
        for it, s in off:
            groups.setdefault(it.topic.split(":")[0].strip() or "other", []).append(s <= OFF_TOPIC_MAX)
        report.by_topic = {g: round(sum(v) / len(v), 3) for g, v in sorted(groups.items())}

    novel = [s for it, s, _ in results.values() if it.label in ("novel", "seq_first")]
    not_novel = [s for it, s, _ in results.values() if it.label in ("paraphrase", "generic")]
    report.auc_novelty = None if auc(novel, not_novel) is None else round(auc(novel, not_novel), 3)
    on = [g for it, _, g in results.values() if it.label in ON_TOPIC]
    off_g = [g for it, _, g in results.values() if it.label == "off_topic"]
    report.auc_relevance = None if auc(on, off_g) is None else round(auc(on, off_g), 3)

    pairs: dict[str, dict[str, EvalItem]] = {}
    for it in items:
        if it.label in ("seq_first", "seq_second") and it.of:
            pairs.setdefault(it.of, {})[it.label] = it
    outcomes = []
    for pair in pairs.values():
        if len(pair) != 2:
            continue
        scorer = base.fork()
        before = scorer.score(pair["seq_second"].submission).score
        scorer.submit(pair["seq_first"].submission)
        after = scorer.score(pair["seq_second"].submission).score
        outcomes.append(before >= NOVEL_MIN and after <= NOT_NOVEL_MAX)
    report.sequential = round(sum(outcomes) / len(outcomes), 3) if outcomes else None
    report.failures.sort(key=lambda f: (f["label"], -f["score"]))
    return report


def format_report(report: EvalReport, failures: int = 10) -> str:
    lines = [f"split: {report.split}   items: {sum(report.counts.values())}", "",
             "| label | n | pass rate |", "|---|---:|---:|"]
    for label, n in sorted(report.counts.items()):
        rate = report.pass_rates.get(label)
        lines.append(f"| {label} | {n} | {'-' if rate is None else f'{rate:.0%}'} |")
    lines.append("")
    if report.off_topic_loose is not None:
        lines.append(f"off-topic at <= {OFF_TOPIC_LOOSE}: {report.off_topic_loose:.0%}   "
                     + "   ".join(f"{g}: {v:.0%}" for g, v in report.by_topic.items()))
    lines.append(f"AUC novel vs not novel: {report.auc_novelty}   AUC relevance (gate): {report.auc_relevance}   "
                 f"sequential: {report.sequential}")
    if failures and report.failures:
        lines += ["", f"worst failures (up to {failures}):"]
        for f in report.failures[:failures]:
            lines.append(f"  {f['label']:<10} {f['score']:.2f} gate {f['relevance_gate']:.2f}  {f['id']}: {f['headline']}")
    return "\n".join(lines)
