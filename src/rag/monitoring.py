"""Token, cost and latency tracking with a daily budget alert.

Logs contain the MASKED question only, never raw PII.
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Callable

# USD per 1M tokens. Illustrative values: update from your provider's pricing page.
PRICING = {
    "extractive-fallback": (0.0, 0.0),
    "catalog": (0.0, 0.0),
    "default": (0.59, 0.79),
}


class CostTracker:
    def __init__(self, log_path: Path, daily_budget_usd: float = 5.0,
                 alert: Callable[[str], None] = print):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.daily_budget_usd = daily_budget_usd
        self.alert = alert

    @staticmethod
    def cost_of(model: str, prompt_tokens: int, completion_tokens: int) -> float:
        in_price, out_price = PRICING.get(model, PRICING["default"])
        return (prompt_tokens * in_price + completion_tokens * out_price) / 1_000_000

    def _entries(self) -> list[dict]:
        if not self.log_path.exists():
            return []
        return [json.loads(line) for line in self.log_path.read_text(encoding="utf-8").splitlines() if line]

    def today_cost(self) -> float:
        today = dt.date.today().isoformat()
        return sum(e["cost_usd"] for e in self._entries() if e["ts"].startswith(today))

    def record(self, answer) -> dict:
        cost = self.cost_of(answer.model, answer.prompt_tokens, answer.completion_tokens)
        entry = {
            "ts": dt.datetime.now(dt.timezone.utc).isoformat(),
            "answer_id": answer.answer_id,
            "user_id": answer.user_id,
            "kind": answer.kind,
            "question": answer.question,
            "refused": answer.refused,
            "refusal_reason": answer.refusal_reason,
            "latency_ms": round(answer.latency_ms, 1),
            "prompt_tokens": answer.prompt_tokens,
            "completion_tokens": answer.completion_tokens,
            "model": answer.model,
            "cost_usd": round(cost, 8),
        }
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
        spent = self.today_cost()
        if spent > self.daily_budget_usd:
            self.alert(f"ALERT: today's LLM spend ${spent:.4f} exceeds budget ${self.daily_budget_usd:.2f}")
        return entry

    def summary(self) -> dict:
        entries = self._entries()
        n = len(entries)
        return {
            "queries": n,
            "refusals": sum(e["refused"] for e in entries),
            "avg_latency_ms": round(sum(e["latency_ms"] for e in entries) / n, 1) if n else 0.0,
            "total_tokens": sum(e["prompt_tokens"] + e["completion_tokens"] for e in entries),
            "total_cost_usd": round(sum(e["cost_usd"] for e in entries), 6),
        }
