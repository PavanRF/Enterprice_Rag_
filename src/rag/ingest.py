"""Document loading, chunking and department tagging.

Every chunk must carry a known department. Documents without one are rejected
at ingestion time, so an untagged file can never become readable by accident.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import docx
from pypdf import PdfReader

SUPPORTED = {".md", ".txt", ".pdf", ".docx"}
_FRONT_MATTER = re.compile(r"\A---\n(.*?)\n---\n", re.S)


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    text: str
    source: str       # path relative to the docs root
    title: str
    department: str


def read_document(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".md", ".txt"}:
        return path.read_text(encoding="utf-8")
    if suffix == ".pdf":
        reader = PdfReader(str(path))
        return "\n\n".join((page.extract_text() or "") for page in reader.pages)
    if suffix == ".docx":
        document = docx.Document(str(path))
        return "\n\n".join(p.text for p in document.paragraphs if p.text.strip())
    raise ValueError(f"unsupported file type: {path}")


def parse_front_matter(text: str) -> tuple[dict[str, str], str]:
    match = _FRONT_MATTER.match(text)
    if not match:
        return {}, text
    meta = {}
    for line in match.group(1).splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            meta[key.strip()] = value.strip()
    return meta, text[match.end():]


def chunk_text(text: str, max_chars: int = 700) -> list[str]:
    """Group paragraphs into chunks of at most max_chars, carrying one paragraph of overlap."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    pieces: list[str] = []
    for p in paragraphs:
        if len(p) <= max_chars:
            pieces.append(p)
        else:
            pieces.extend(p[i:i + max_chars] for i in range(0, len(p), max_chars))

    chunks: list[str] = []
    current: list[str] = []
    size = 0
    for piece in pieces:
        if current and size + len(piece) > max_chars:
            chunks.append("\n\n".join(current))
            current = current[-1:]
            size = len(current[0]) if current else 0
        current.append(piece)
        size += len(piece)
    if current:
        chunks.append("\n\n".join(current))
    return chunks


def ingest_directory(root: Path, departments: Iterable[str], max_chars: int = 700) -> list[Chunk]:
    """Load every supported file under root.

    Department comes from front matter ("department: finance") or, failing that,
    the name of the folder the file sits in. Unknown or missing departments raise.
    """
    root = Path(root)
    known = set(departments)
    chunks: list[Chunk] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED:
            continue
        meta, body = parse_front_matter(read_document(path))
        relative = path.relative_to(root)
        department = meta.get("department") or (
            relative.parts[0] if len(relative.parts) > 1 else None
        )
        if department not in known:
            raise ValueError(
                f"{relative}: department {department!r} is missing or unknown. "
                "Refusing to ingest untagged documents (fail closed)."
            )
        title = meta.get("title", path.stem.replace("_", " ").title())
        for i, piece in enumerate(chunk_text(body, max_chars)):
            chunks.append(Chunk(
                chunk_id=f"{path.stem}-{i:03d}",
                text=piece,
                source=relative.as_posix(),
                title=title,
                department=department,
            ))
    return chunks
