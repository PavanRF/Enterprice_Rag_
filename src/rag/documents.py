"""Document upload, deletion and listing, enforced by role.

Every operation checks the user's WRITE permission for the target department.
Paths are validated so a crafted filename cannot escape the documents folder.
"""
from __future__ import annotations

import datetime as dt
import os
import re
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from .guardrails import mask_pii
from .ingest import SUPPORTED, parse_front_matter, read_document
from .rbac import AccessPolicy, User

MAX_BYTES_DEFAULT = 10 * 1024 * 1024


class DocumentError(Exception):
    pass


@dataclass
class SaveResult:
    source: str
    department: str
    title: str
    pii_warnings: list[str] = field(default_factory=list)


def docs_root(root: Path) -> Path:
    return Path(root) / "data" / "docs"


def safe_filename(name: str) -> str:
    base = Path(name.replace("\\", "/")).name            # drop any folder parts
    stem, suffix = os.path.splitext(base)
    stem = re.sub(r"[^A-Za-z0-9_-]+", "_", stem).strip("_") or "document"
    return stem + suffix.lower()


def _check_inside(path: Path, root: Path) -> Path:
    resolved = path.resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise DocumentError("invalid document path")
    return resolved


def save_document(root: Path, policy: AccessPolicy, user: User, department: str,
                  filename: str, data: bytes, replace: bool = False,
                  max_bytes: int = MAX_BYTES_DEFAULT) -> SaveResult:
    if department not in policy.writable_departments(user.roles):
        raise DocumentError(f"you do not have permission to upload to {department}")
    name = safe_filename(filename)
    if Path(name).suffix not in SUPPORTED:
        raise DocumentError("unsupported file type. Use PDF, DOCX, MD or TXT.")
    if not data:
        raise DocumentError("the file is empty")
    if len(data) > max_bytes:
        raise DocumentError(f"file is larger than {max_bytes // (1024 * 1024)} MB")

    base = docs_root(root)
    dest = _check_inside(base / department / name, base)
    if dest.exists() and not replace:
        raise DocumentError(f"'{name}' already exists in {department}. Tick 'replace' to overwrite it.")

    # Validate the content in a temporary file outside the documents folder,
    # so a broken or empty file is never indexed.
    fd, tmp_name = tempfile.mkstemp(suffix=Path(name).suffix)
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        tmp.write_bytes(data)
        text = read_document(tmp)
    except Exception as error:
        tmp.unlink(missing_ok=True)
        raise DocumentError(f"could not read the file: {error}") from None
    _, body = parse_front_matter(text)
    if not body.strip():
        tmp.unlink(missing_ok=True)
        raise DocumentError("no readable text found (scanned PDFs need OCR first)")

    _, found = mask_pii(body)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(tmp), str(dest))
    title = Path(name).stem.replace("_", " ").replace("-", " ").title()
    return SaveResult(
        source=f"{department}/{name}",
        department=department,
        title=title,
        pii_warnings=sorted(set(found)),
    )


def delete_document(root: Path, policy: AccessPolicy, user: User, source: str) -> None:
    parts = Path(source).parts
    if len(parts) != 2:
        raise DocumentError("invalid document")
    department = parts[0]
    if department not in policy.writable_departments(user.roles):
        raise DocumentError(f"you do not have permission to delete {department} documents")
    path = _check_inside(docs_root(root) / source, docs_root(root))
    if not path.is_file():
        raise DocumentError("document not found")
    path.unlink()


def list_documents(root: Path, policy: AccessPolicy, user: User) -> list[dict]:
    base = docs_root(root)
    readable = policy.allowed_departments(user.roles)
    writable = policy.writable_departments(user.roles)
    items = []
    for path in sorted(base.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED:
            continue
        relative = path.relative_to(base)
        if len(relative.parts) != 2:
            continue
        department = relative.parts[0]
        if department not in readable:
            continue  # restricted documents are not even listed
        meta, _ = parse_front_matter(read_document(path))
        stat = path.stat()
        items.append({
            "title": meta.get("title", path.stem.replace("_", " ").title()),
            "department": department,
            "source": relative.as_posix(),
            "size_kb": round(stat.st_size / 1024, 1),
            "updated": dt.datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M"),
            "can_delete": department in writable,
        })
    return items
