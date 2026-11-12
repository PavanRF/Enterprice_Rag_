"""Usage analytics for the executive dashboard.

Question text is NOT included in the summary. Employees' questions can be
sensitive (for example, about their own pay), so the dashboard shows counts only.
"""
from __future__ import annotations

import datetime as dt
import json
from collections import Counter
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def summarize(queries: list[dict], feedback: list[dict], days: int = 14,
              today: dt.date | None = None) -> dict:
    today = today or dt.date.today()
    total = len(queries)
    refused = sum(1 for q in queries if q.get("refused"))
    reasons = Counter(q["refusal_reason"] for q in queries if q.get("refusal_reason"))
    kinds = Counter(q.get("kind", "answer") for q in queries)
    per_day = Counter(q["ts"][:10] for q in queries)
    daily = []
    for offset in range(days - 1, -1, -1):
        day = (today - dt.timedelta(days=offset)).isoformat()
        daily.append({"day": day, "queries": per_day.get(day, 0)})
    up = sum(1 for f in feedback if f.get("helpful"))
    return {
        "queries": total,
        "refusals": refused,
        "refusal_rate": round(refused / total, 3) if total else 0.0,
        "avg_latency_ms": round(sum(q.get("latency_ms", 0) for q in queries) / total, 1) if total else 0.0,
        "total_cost_usd": round(sum(q.get("cost_usd", 0) for q in queries), 6),
        "total_tokens": sum(q.get("prompt_tokens", 0) + q.get("completion_tokens", 0) for q in queries),
        "refusal_reasons": dict(reasons),
        "answer_kinds": dict(kinds),
        "daily": daily,
        "feedback_up": up,
        "feedback_down": len(feedback) - up,
    }
