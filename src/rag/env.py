"""Load KEY=VALUE settings from a .env file into the environment.

Real environment variables always win, so production secrets set by the
platform are never overridden by a file. Standard library only.
"""
from __future__ import annotations

import os
from pathlib import Path


def load_dotenv(path: Path) -> dict[str, str]:
    path = Path(path)
    if not path.exists():
        return {}
    loaded: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.split(" #", 1)[0].strip().strip('"').strip("'")
        key = key.strip()
        if key and key not in os.environ:
            os.environ[key] = value
            loaded[key] = value
    return loaded
