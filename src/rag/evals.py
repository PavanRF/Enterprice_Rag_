"""Evaluation harness: runs a labelled question set through the pipeline and gates on thresholds.

Metrics:
  refusal_accuracy  - answered vs refused matches the expected behaviour
  keyword_recall    - required facts appear in the answer
  source_hit_rate   - the expected document was cited
  rbac_leaks        - restricted text or sources appeared for a user without access (must be 0)
  pii_leaks         - raw PII appeared in an answer or the logged question (must be 0)
"""
from __future__ import annotations

from pathlib import Path

import yaml


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 1.0


def run_evals(pipeline, users, cases: list[dict], thresholds: dict) -> dict:
    refusal_ok, keyword_scores, source_hits = [], [], []
    rbac_leaks, pii_leaks = 0, 0
    failures: list[dict] = []

    for case in cases:
        user = users.get(case["user"])
        answer = pipeline.answer(user, case["question"])
        text = answer.text.lower()
        cited_sources = [c.source for c in answer.citations]
        problems: list[str] = []

        expect = case.get("expect", "any")
        if expect in ("answer", "refuse"):
            correct = answer.refused == (expect == "refuse")
            refusal_ok.append(correct)
            if not correct:
                problems.append(f"expected {expect}, got {'refuse' if answer.refused else 'answer'}")

        must_contain = case.get("must_contain", [])
        if must_contain:
            score = 0.0 if answer.refused else (
                sum(k.lower() in text for k in must_contain) / len(must_contain))
            keyword_scores.append(score)
            if score < 1.0:
                problems.append(f"missing facts: {must_contain}")

        if case.get("expected_source"):
            hit = case["expected_source"] in cited_sources
            source_hits.append(hit)
            if not hit:
                problems.append(f"expected source {case['expected_source']} not cited")

        restricted_phrases = [p.lower() for p in case.get("restricted_phrases", [])]
        restricted_sources = case.get("restricted_sources", [])
        leaked = (any(p in text for p in restricted_phrases)
                  or any(src.endswith(r) for src in cited_sources for r in restricted_sources))
        if leaked:
            rbac_leaks += 1
            problems.append("RBAC LEAK: restricted content reached this user")

        forbidden = [f.lower() for f in case.get("forbidden_text", [])]
        visible = (text + " " + answer.question.lower())
        if any(f in visible for f in forbidden):
            pii_leaks += 1
            problems.append("PII LEAK: raw sensitive value visible")

        if problems:
            failures.append({"id": case["id"], "question": case["question"], "problems": problems})

    metrics = {
        "cases": len(cases),
        "refusal_accuracy": round(_mean([float(x) for x in refusal_ok]), 3),
        "keyword_recall": round(_mean(keyword_scores), 3),
        "source_hit_rate": round(_mean([float(x) for x in source_hits]), 3),
        "rbac_leaks": rbac_leaks,
        "pii_leaks": pii_leaks,
    }
    checks = {
        "refusal_accuracy": metrics["refusal_accuracy"] >= thresholds["refusal_accuracy"],
        "keyword_recall": metrics["keyword_recall"] >= thresholds["keyword_recall"],
        "source_hit_rate": metrics["source_hit_rate"] >= thresholds["source_hit_rate"],
        "rbac_leaks": metrics["rbac_leaks"] <= thresholds["rbac_leaks"],
        "pii_leaks": metrics["pii_leaks"] <= thresholds["pii_leaks"],
    }
    return {"metrics": metrics, "checks": checks, "passed": all(checks.values()), "failures": failures}


def load_testset(path: Path) -> list[dict]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))
