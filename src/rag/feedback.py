"""Thumbs up/down on answers. Stored without user-typed text, so nothing sensitive is copied."""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path


class FeedbackLog:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, user_id: str, answer_id: str, helpful: bool) -> None:
        entry = {
            "ts": dt.datetime.now(dt.timezone.utc).isoformat(),
            "user_id": user_id,
            "answer_id": answer_id,
            "helpful": bool(helpful),
        }
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

    def entries(self) -> list[dict]:
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines() if line]
