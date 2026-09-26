"""CLI.

    python -m novelty demo                      # score every labelled probe
    python -m novelty corpus                    # leave-one-out novelty of each corpus item
    python -m novelty score --headline ... --body ... --stance support [--json]
    python -m novelty serve [--port 8000]      # minimal web UI at http://127.0.0.1:8000

Global options: --log-level LEVEL, or -v for DEBUG. Logs go to stderr (and NOVELTY_LOG_FILE if
set). ``score`` and ``serve`` log at INFO by default; ``demo`` and ``corpus`` only warnings, so
their tables stay readable. Exit codes: 0 ok (including ``serve`` stopped with Ctrl+C), 2 a
reported error or bad arguments, 130 interrupted.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import os
import sys

from .data import build_scorer, load_probes
from .errors import NoveltyError
from .logging_setup import configure_logging
from .models import Stance, Submission

log = logging.getLogger("novelty.cli")


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


def _cmd_serve(args: argparse.Namespace) -> None:
    from .server import serve

    serve(port=args.port)


_DEFAULT_LEVEL = {"demo": "WARNING", "corpus": "WARNING", "score": "INFO", "serve": "INFO"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="novelty")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], type=str.upper,
                        help="default: INFO for score/serve, WARNING for demo/corpus (or NOVELTY_LOG_LEVEL)")
    parser.add_argument("-v", "--verbose", action="store_true", help="shorthand for --log-level DEBUG")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("demo").set_defaults(fn=_cmd_demo)
    sub.add_parser("corpus").set_defaults(fn=_cmd_corpus)
    p = sub.add_parser("score")
    p.add_argument("--headline", required=True)
    p.add_argument("--body", required=True)
    p.add_argument("--stance", required=True, choices=[s.value for s in Stance])
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=_cmd_score)
    p = sub.add_parser("serve")
    p.add_argument("--port", type=int, default=8000)
    p.set_defaults(fn=_cmd_serve)
    args = parser.parse_args(argv)

    level = "DEBUG" if args.verbose else (args.log_level or os.environ.get("NOVELTY_LOG_LEVEL")
                                          or _DEFAULT_LEVEL[args.cmd])
    configure_logging(level)
    try:
        args.fn(args)
    except NoveltyError as e:
        log.debug("command failed", exc_info=True)
        print(f"error: {e}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
