"""Wire the components together from config files and the environment."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import yaml

from .env import load_dotenv
from .ingest import ingest_directory
from .llm import LLM, llm_from_env
from .monitoring import CostTracker
from .pipeline import RAGPipeline
from .rbac import load_access_config
from .store import TfidfStore


def _yaml(root: Path, name: str) -> dict:
    return yaml.safe_load((Path(root) / "config" / name).read_text(encoding="utf-8"))


def load_settings(root: Path) -> dict:
    return _yaml(root, "settings.yaml")


def load_branding(root: Path) -> dict:
    return _yaml(root, "branding.yaml")


def credentials_path(root: Path) -> Path:
    return Path(root) / "data" / "auth" / "credentials.json"


def source_signature(root: Path) -> str:
    """Changes whenever a document or config file is added, edited or removed."""
    root = Path(root)
    digest = hashlib.sha256()
    files = sorted((root / "data/docs").rglob("*")) + sorted((root / "config").glob("*.yaml"))
    for path in files:
        if path.is_file():
            stat = path.stat()
            digest.update(f"{path.relative_to(root)}|{stat.st_mtime_ns}|{stat.st_size}".encode())
    return digest.hexdigest()


def llm_mode() -> str:
    if os.environ.get("GROQ_API_KEY"):
        return f"Groq · {os.environ.get('GROQ_MODEL', 'openai/gpt-oss-120b')}"
    return "No LLM key set: extractive fallback"


def build_pipeline(root: Path, llm: LLM | None = None, log_path: Path | None = None):
    """Returns (pipeline, user_directory). Pipeline.policy holds the access policy."""
    root = Path(root)
    load_dotenv(root / ".env")
    settings = load_settings(root)
    policy, users = load_access_config(root / "config/access.yaml")

    chunks = ingest_directory(root / "data/docs", policy.departments,
                              max_chars=settings["ingestion"]["max_chunk_chars"])
    store = TfidfStore()
    store.add(chunks)

    tracker = CostTracker(
        log_path or root / settings["monitoring"]["log_path"],
        daily_budget_usd=float(os.environ.get("DAILY_BUDGET_USD", settings["monitoring"]["daily_budget_usd"])),
    )
    pipeline = RAGPipeline(
        store, policy, llm or llm_from_env(), tracker=tracker,
        top_k=settings["retrieval"]["top_k"],
        min_score=settings["retrieval"]["min_score"],
        history_turns=settings["retrieval"]["history_turns"],
    )
    return pipeline, users
