"""CLI.

    python -m novelty demo                      # score every labelled probe
    python -m novelty corpus                    # leave-one-out novelty of each corpus item
    python -m novelty score --headline ... --body ... --stance support [--json]
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys

from .data import build_scorer, load_probes
from .models import Stance, Submission


def _cmd_demo(_: argparse.Namespace) -> None:
    scorer = build_scorer()
    print(f"embedder: {scorer.embedder.name}   corpus: {len(scorer.corpus)} submissions\n")
    print(f"{'group':<15} {'id':<22} {'score':>6} {'novelty':>8} {'relev.':>7} {'gate':>5}  note")
    for group, probes in load_probes().items():
        for p in probes:
            r = scorer.score(Submission.from_dict(p))
            note = r.reasons[-1] if r.reasons else ""
            print(f"{group:<15} {p['id']:<22} {r.score:>6.3f} {r.novelty:>8.3f} {r.relevance:>7.3f} {r.relevance_gate:>5.2f}  {note}")


def _cmd_corpus(_: argparse.Namespace) -> None:
    scorer = build_scorer()
    subs = {s.id: s for s in scorer.corpus}
    for cid, nov in sorted(scorer.corpus_novelty().items(), key=lambda kv: -kv[1]):
        print(f"{cid}  {nov:.3f}  {subs[cid].headline}")


def _cmd_score(args: argparse.Namespace) -> None:
    sub = Submission(headline=args.headline, body=args.body, stance=Stance(args.stance))
    r = build_scorer().score(sub)
    if args.json:
        print(json.dumps(dataclasses.asdict(r), indent=2))
        return
    print(f"score {r.score:.3f}  (novelty {r.novelty:.3f} x relevance gate {r.relevance_gate:.2f})")
    for reason in r.reasons:
        print(f"  - {reason}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="novelty")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("demo").set_defaults(fn=_cmd_demo)
    sub.add_parser("corpus").set_defaults(fn=_cmd_corpus)
    p = sub.add_parser("score")
    p.add_argument("--headline", required=True)
    p.add_argument("--body", required=True)
    p.add_argument("--stance", required=True, choices=[s.value for s in Stance])
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=_cmd_score)
    args = parser.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
