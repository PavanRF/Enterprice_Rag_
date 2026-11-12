"""In-memory TF-IDF vector store with RBAC pre-filtering.

Access is enforced BEFORE scoring: chunks the caller may not read are never
candidates, so they cannot be ranked, returned, or cited. The same interface
can be backed by Qdrant, Chroma or Milvus.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

from .ingest import Chunk

_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = set(ENGLISH_STOP_WORDS) | {"tell", "please", "what", "how", "much", "does", "many"}
# Lightweight domain synonyms. Dense embeddings handle paraphrase in general;
# this keeps the lexical fallback useful for the common cases.
_SYNONYMS = {"acquisition": "acquire", "acquiring": "acquire", "acquired": "acquire", "acquires": "acquire"}


def tokenize(text: str) -> list[str]:
    tokens = []
    for tok in _TOKEN.findall(text.lower()):
        if tok in _STOP:
            continue
        tok = _SYNONYMS.get(tok, tok)
        if len(tok) > 3 and tok.endswith("s") and not tok.endswith("ss"):
            tok = tok[:-1]  # crude plural stemming: passwords -> password, meals -> meal
        tokens.append(tok)
    return tokens


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float


class TfidfStore:
    def __init__(self) -> None:
        self._chunks: list[Chunk] = []
        self._vectorizer: TfidfVectorizer | None = None
        self._matrix = None

    def __len__(self) -> int:
        return len(self._chunks)

    @property
    def chunks(self) -> list[Chunk]:
        return list(self._chunks)

    def add(self, chunks: Iterable[Chunk]) -> None:
        self._chunks.extend(chunks)
        self._vectorizer = None
        self._matrix = None

    def _fit(self) -> None:
        self._vectorizer = TfidfVectorizer(tokenizer=tokenize, token_pattern=None, lowercase=False,
                                          ngram_range=(1, 2), sublinear_tf=True)
        # Index the title with the text: a question naming a document's title now matches it.
        self._matrix = self._vectorizer.fit_transform([f"{c.title}. {c.text}" for c in self._chunks])

    def search(self, query: str, allowed: frozenset[str], k: int = 4) -> list[Hit]:
        candidates = [i for i, c in enumerate(self._chunks) if c.department in allowed]
        if not candidates:
            return []
        if self._vectorizer is None:
            self._fit()
        q = self._vectorizer.transform([query])
        if q.nnz == 0:
            return []
        sims = (self._matrix[candidates] @ q.T).toarray().ravel()
        order = np.argsort(-sims)[:k]
        return [Hit(self._chunks[candidates[i]], float(sims[i])) for i in order if sims[i] > 0]
