"""The RAG request flow.

guardrails -> catalog shortcut or follow-up rewrite -> RBAC-filtered retrieval
-> grounded generation -> output guardrails -> log.
"""
from __future__ import annotations

import re
import time
import uuid
from dataclasses import dataclass, field

from .guardrails import is_out_of_scope, is_prompt_injection, mask_pii
from .llm import LLM
from .rbac import AccessPolicy, User
from .store import TfidfStore

REFUSAL_INJECTION = "I can't process that request."
REFUSAL_OUT_OF_SCOPE = (
    "That's outside what this assistant can help with. Try asking about company "
    "policies, finance, HR, marketing or strategy documents you have access to."
)
# Deliberately identical whether no document matched or matching documents are restricted,
# so the response never reveals that a restricted document exists.
REFUSAL_NO_ANSWER = (
    "I couldn't find an answer in the documents available to you. Try a more specific "
    "question, or ask 'what documents are available?' to see what you can search."
)

_CATALOG_ASK = re.compile(
    # "what documents...", "list my files", "what are the available documents"
    r"\b(what|which|list|show)\s+(?:\w+\s+){0,3}?(documents?|docs?|files?)\b"
    # "available documents", "documents I can access"
    r"|\b(available|accessible)\s+(?:\w+\s+){0,2}?(documents?|docs?|files?)\b"
    r"|\b(documents?|docs?|files?)\s+(?:\w+\s+){0,3}?(available|accessible)\b",
    re.I,
)
# Topic questions about documents ("which document mentions invoices") are real searches.
_TOPIC_WORDS = re.compile(r"\b(about|mention\w*|contain\w*|say|says|cover\w*|regarding|on)\b", re.I)


def is_catalog_question(text: str) -> bool:
    return bool(_CATALOG_ASK.search(text)) and not _TOPIC_WORDS.search(text)


@dataclass(frozen=True)
class Citation:
    n: int
    source: str
    title: str
    department: str
    chunk_id: str


@dataclass
class Answer:
    user_id: str
    question: str                 # masked question, safe to log and display
    text: str
    kind: str = "answer"          # answer | catalog | refusal
    citations: list[Citation] = field(default_factory=list)
    refused: bool = False
    refusal_reason: str | None = None
    pii_found: list[str] = field(default_factory=list)
    rewritten_question: str | None = None
    latency_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    model: str = ""
    answer_id: str = field(default_factory=lambda: uuid.uuid4().hex)


class RAGPipeline:
    def __init__(self, store: TfidfStore, policy: AccessPolicy, llm: LLM, tracker=None,
                 top_k: int = 6, min_score: float = 0.1, history_turns: int = 6):
        self.store = store
        self.policy = policy
        self.llm = llm
        self.tracker = tracker
        self.top_k = top_k
        self.min_score = min_score
        self.history_turns = history_turns

    # ----- catalog helpers -----

    def catalog(self, user: User) -> list[dict]:
        allowed = self.policy.allowed_departments(user.roles)
        docs: dict[str, dict] = {}
        for chunk in self.store.chunks:
            if chunk.department not in allowed:
                continue
            entry = docs.setdefault(chunk.source, {
                "title": chunk.title, "department": chunk.department,
                "source": chunk.source, "chunks": 0,
            })
            entry["chunks"] += 1
        return sorted(docs.values(), key=lambda d: (d["department"], d["title"]))

    def suggestions(self, user: User, limit: int = 4) -> list[str]:
        titles = [d["title"] for d in self.catalog(user)]
        picks = [f"Summarize the {t}" for t in titles[:limit]]
        if not picks:
            return ["What documents are available?"]
        return picks

    # ----- main entry point -----

    def answer(self, user: User, question: str, history: list[dict] | None = None) -> Answer:
        started = time.perf_counter()
        masked_question, pii_in_question = mask_pii(question)

        if is_prompt_injection(masked_question):
            return self._finish(user, masked_question, pii_in_question, started,
                                REFUSAL_INJECTION, "injection")
        if is_out_of_scope(masked_question):
            return self._finish(user, masked_question, pii_in_question, started,
                                REFUSAL_OUT_OF_SCOPE, "out_of_scope")

        if is_catalog_question(masked_question):
            return self._catalog_answer(user, masked_question, pii_in_question, started)

        allowed = self.policy.allowed_departments(user.roles)
        prompt_tokens = completion_tokens = 0
        search_question = masked_question
        rewritten = None

        recent = [{"role": m["role"], "content": mask_pii(m["content"])[0]}
                  for m in (history or [])[-self.history_turns:]]
        if recent:
            rewrite = self.llm.rewrite(masked_question, recent)
            prompt_tokens += rewrite.prompt_tokens
            completion_tokens += rewrite.completion_tokens
            if rewrite.text.strip() and rewrite.text.strip() != masked_question:
                rewritten = rewrite.text.strip()
                search_question = rewritten

        hits = [h for h in self.store.search(search_question, allowed, self.top_k)
                if h.score >= self.min_score]

        # Defense in depth: the store already filtered by role. Verify it anyway.
        if any(h.chunk.department not in allowed for h in hits):
            raise RuntimeError("RBAC violation: retrieval returned a chunk the user cannot read")
        if not hits:
            return self._finish(user, masked_question, pii_in_question, started,
                                REFUSAL_NO_ANSWER, "no_grounded_answer",
                                rewritten, prompt_tokens, completion_tokens)

        result = self.llm.complete(search_question, [h.chunk.text for h in hits])
        prompt_tokens += result.prompt_tokens
        completion_tokens += result.completion_tokens
        if not result.answerable:
            return self._finish(user, masked_question, pii_in_question, started,
                                REFUSAL_NO_ANSWER, "no_grounded_answer",
                                rewritten, prompt_tokens, completion_tokens)

        text, pii_in_output = mask_pii(result.text)
        cited = {int(n) for n in re.findall(r"\[(\d+)\]", text)}
        all_citations = [
            Citation(n, h.chunk.source, h.chunk.title, h.chunk.department, h.chunk.chunk_id)
            for n, h in enumerate(hits, 1)
        ]
        citations = [c for c in all_citations if c.n in cited] or all_citations
        answer = Answer(
            user_id=user.user_id,
            question=masked_question,
            text=text,
            kind="answer",
            citations=citations,
            pii_found=pii_in_question + pii_in_output,
            rewritten_question=rewritten,
            latency_ms=(time.perf_counter() - started) * 1000,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            model=result.model,
        )
        return self._log(answer)

    # ----- internals -----

    def _catalog_answer(self, user, masked_question, pii, started) -> Answer:
        docs = self.catalog(user)
        if not docs:
            text = "No documents are available to your role yet."
        else:
            lines = [f"You can search **{len(docs)} documents**:"]
            lines += [f"- **{d['title']}** ({d['department']})" for d in docs]
            text = "\n".join(lines)
        answer = Answer(
            user_id=user.user_id, question=masked_question, text=text, kind="catalog",
            pii_found=pii, latency_ms=(time.perf_counter() - started) * 1000, model="catalog",
        )
        return self._log(answer)

    def _finish(self, user, masked_question, pii, started, text, reason,
                rewritten=None, prompt_tokens=0, completion_tokens=0) -> Answer:
        answer = Answer(
            user_id=user.user_id,
            question=masked_question,
            text=text,
            kind="refusal",
            refused=True,
            refusal_reason=reason,
            pii_found=pii,
            rewritten_question=rewritten,
            latency_ms=(time.perf_counter() - started) * 1000,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            model=self.llm.model,
        )
        return self._log(answer)

    def _log(self, answer: Answer) -> Answer:
        if self.tracker is not None:
            self.tracker.record(answer)
        return answer
