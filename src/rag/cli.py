"""Command line.

  python -m rag.cli ingest                       show what is indexed, per department
  python -m rag.cli set-password <user_id>       set or change a user's password (prompted)
  python -m rag.cli ask --user <id> "question"   ask as a user
  python -m rag.cli eval                         run the quality and security gate
  python -m rag.cli stats                        query log summary
"""
from __future__ import annotations

import argparse
import getpass
import json
import sys
from collections import Counter
from pathlib import Path

from .auth import AuthError, set_password
from .evals import load_testset, run_evals
from .factory import build_pipeline, credentials_path, load_settings
from .ingest import ingest_directory
from .rbac import load_access_config

ROOT = Path(__file__).resolve().parents[2]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="rag")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("ingest")
    pw = sub.add_parser("set-password")
    pw.add_argument("user_id")
    ask = sub.add_parser("ask")
    ask.add_argument("--user", required=True)
    ask.add_argument("question")
    ev = sub.add_parser("eval")
    ev.add_argument("--testset", default=str(ROOT / "evals/testset.yaml"))
    ev.add_argument("--report", default=None)
    sub.add_parser("stats")
    args = parser.parse_args(argv)

    if args.cmd == "ingest":
        settings = load_settings(ROOT)
        policy, _ = load_access_config(ROOT / "config/access.yaml")
        chunks = ingest_directory(ROOT / "data/docs", policy.departments,
                                  max_chars=settings["ingestion"]["max_chunk_chars"])
        counts = Counter(c.department for c in chunks)
        print(f"{len(chunks)} chunks indexed")
        for dept, n in sorted(counts.items()):
            print(f"  {dept:<10} {n}")
        return 0

    if args.cmd == "set-password":
        _, users = load_access_config(ROOT / "config/access.yaml")
        if args.user_id not in {u.user_id for u in users.all()}:
            print(f"unknown user '{args.user_id}'. Add them to config/access.yaml first.")
            return 1
        password = getpass.getpass(f"New password for {args.user_id}: ")
        if password != getpass.getpass("Repeat password: "):
            print("passwords do not match")
            return 1
        try:
            set_password(credentials_path(ROOT), args.user_id, password,
                         min_length=load_settings(ROOT)["auth"]["min_password_length"])
        except AuthError as error:
            print(error)
            return 1
        print(f"password set for {args.user_id}")
        return 0

    pipeline, users = build_pipeline(ROOT)

    if args.cmd == "ask":
        answer = pipeline.answer(users.get(args.user), args.question)
        print(answer.text)
        if answer.citations:
            print("\nSources:")
            for c in answer.citations:
                print(f"  [{c.n}] {c.title} ({c.source})")
        if answer.refused:
            print(f"\n(refused: {answer.refusal_reason})")
        return 0

    if args.cmd == "eval":
        thresholds = load_settings(ROOT)["eval_thresholds"]
        report = run_evals(pipeline, users, load_testset(Path(args.testset)), thresholds)
        print(json.dumps(report["metrics"], indent=2))
        for name, ok in report["checks"].items():
            print(f"  {'PASS' if ok else 'FAIL'}  {name}")
        for f in report["failures"]:
            print(f"  - {f['id']}: {'; '.join(f['problems'])}")
        if args.report:
            Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print("\nGATE:", "PASSED" if report["passed"] else "FAILED")
        return 0 if report["passed"] else 1

    if args.cmd == "stats":
        print(json.dumps(pipeline.tracker.summary(), indent=2))
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
