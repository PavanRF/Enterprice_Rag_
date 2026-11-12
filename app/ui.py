"""Streamlit screens. Business logic lives in src/rag; this module renders it."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import streamlit as st

from rag.analytics import load_jsonl, summarize
from rag.auth import AuthError, authenticate, load_credentials
from rag.documents import DocumentError, delete_document, list_documents, save_document
from rag.feedback import FeedbackLog
from rag.llm import LLMError
from rag.presentation import background_options, citation_label, export_markdown, theme_css

MB = 1024 * 1024


@dataclass
class Ctx:
    root: Path
    pipeline: object
    users: object
    user: object
    branding: dict
    settings: dict
    feedback: FeedbackLog
    creds_file: Path
    limiter: object
    llm: str
    auth_mode: str = "login"


def apply_theme(branding: dict, choice: str, root: Path) -> None:
    st.markdown(theme_css(branding, choice, root), unsafe_allow_html=True)


def hero(branding: dict) -> None:
    st.markdown(f"<div class='hero'><h1>{branding['app_name']}</h1>"
                f"<p>{branding.get('tagline', '')}</p></div>", unsafe_allow_html=True)


# ---------- login ----------

def render_login(branding: dict, users, creds_file: Path, limiter, root: Path) -> None:
    hero(branding)
    options = background_options(branding, root)
    default = branding.get("default_background")
    _, middle, _ = st.columns([1, 1.2, 1])
    with middle:
        st.selectbox("Background", options, index=options.index(default) if default in options else 0,
                     key="bg", help="Pick a look for the app.")
        with st.form("login_form", clear_on_submit=False):
            username = st.text_input("Username", placeholder="e.g. priya.sharma")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Sign in", type="primary", use_container_width=True)
        if submitted:
            try:
                user = authenticate(users, load_credentials(creds_file),
                                    username.strip(), password, limiter)
            except AuthError as error:
                st.error(str(error))
            else:
                st.session_state["user_id"] = user.user_id
                st.rerun()


def render_google_login(branding: dict) -> None:
    hero(branding)
    _, middle, _ = st.columns([1, 1.2, 1])
    with middle:
        st.markdown("Sign in with your company Google account to continue.")
        if st.button("Sign in with Google", type="primary", use_container_width=True):
            st.login()


# ---------- sidebar ----------

def render_sidebar(ctx: Ctx) -> None:
    user = ctx.user
    policy = ctx.pipeline.policy
    st.markdown(f"**{user.name}**")
    st.caption(f"Roles: {', '.join(user.roles)}")
    st.caption(f"Can read: {', '.join(sorted(policy.allowed_departments(user.roles)))}")
    st.caption(f"Answering with: {ctx.llm}")
    st.caption(f"Indexed chunks: {len(ctx.pipeline.store)}")
    st.selectbox("Background", background_options(ctx.branding, ctx.root), key="bg")
    if st.button("Sign out", use_container_width=True):
        if ctx.auth_mode == "google":
            st.logout()  # also ends the Google session in the browser
        st.session_state.clear()
        st.rerun()


# ---------- assistant ----------

def _messages(ctx: Ctx) -> list[dict]:
    return st.session_state.setdefault(f"messages:{ctx.user.user_id}", [])


def page_assistant() -> None:
    ctx: Ctx = st.session_state["ctx"]
    messages = _messages(ctx)
    st.subheader("Ask your documents")
    st.caption("Answers come only from documents your role can read, with sources.")

    bar = st.columns([4, 1, 1])
    with bar[1]:
        if st.button("Clear chat", use_container_width=True, disabled=not messages):
            messages.clear()
            st.rerun()
    with bar[2]:
        st.download_button("Export chat", data=export_markdown(messages, ctx.user.name),
                           file_name="chat-export.md", mime="text/markdown",
                           use_container_width=True, disabled=not messages)

    if not messages:
        st.caption("Try one of these:")
        cols = st.columns(2)
        for i, suggestion in enumerate(ctx.pipeline.suggestions(ctx.user)):
            with cols[i % 2]:
                if st.button(suggestion, key=f"sugg-{i}", use_container_width=True):
                    st.session_state["pending"] = suggestion
                    st.rerun()

    for m in messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if m["role"] != "assistant":
                continue
            if m.get("citations"):
                with st.expander(f"Sources ({len(m['citations'])})"):
                    for c in m["citations"]:
                        st.caption(citation_label(c))
            if m.get("pii"):
                st.caption("Note: personal data in your question was masked.")
            if m.get("kind") == "answer":
                value = st.feedback("thumbs", key=f"fb-{m['answer_id']}")
                if value is not None and value != m.get("rating"):
                    ctx.feedback.record(ctx.user.user_id, m["answer_id"], helpful=(value == 1))
                    m["rating"] = value
                    st.toast("Thanks, your feedback helps improve answers.")

    typed = st.chat_input("Ask about a document you can access")
    question = typed or st.session_state.pop("pending", None)
    if not question:
        return

    history = [{"role": m["role"], "content": m["content"]} for m in messages]
    try:
        answer = ctx.pipeline.answer(ctx.user, question, history=history)
    except LLMError as error:
        st.error(f"The AI service did not respond ({error}). Please try again in a moment.")
        return

    messages.append({"role": "user", "content": question})
    messages.append({
        "role": "assistant",
        "content": answer.text,
        "kind": answer.kind,
        "answer_id": answer.answer_id,
        "refused": answer.refused,
        "pii": bool(answer.pii_found),
        "citations": [{"n": c.n, "title": c.title, "department": c.department, "source": c.source}
                      for c in answer.citations],
    })
    st.rerun()


# ---------- documents ----------

def page_documents() -> None:
    ctx: Ctx = st.session_state["ctx"]
    st.subheader("Document library")
    st.caption("Only documents your role can read are listed.")
    rows = list_documents(ctx.root, ctx.pipeline.policy, ctx.user)
    if not rows:
        st.info("No documents are available to your role yet.")
        return
    frame = pd.DataFrame(rows)
    query = st.text_input("Search by title", placeholder="e.g. leave")
    if query:
        frame = frame[frame["title"].str.contains(query, case=False)]
    st.dataframe(frame[["title", "department", "size_kb", "updated", "source"]],
                 use_container_width=True, hide_index=True)

    deletable = [r["source"] for r in rows if r["can_delete"]]
    if deletable:
        st.markdown("#### Remove a document")
        choice = st.selectbox("Document", deletable, key="delete_choice")
        confirmed = st.checkbox(f"I understand '{choice}' will be removed from search for everyone.")
        if st.button("Delete document", type="primary", disabled=not confirmed):
            try:
                delete_document(ctx.root, ctx.pipeline.policy, ctx.user, choice)
            except DocumentError as error:
                st.error(str(error))
            else:
                st.success(f"Removed {choice}.")
                st.rerun()


# ---------- upload ----------

def page_upload() -> None:
    ctx: Ctx = st.session_state["ctx"]
    st.subheader("Upload a document")
    writable = sorted(ctx.pipeline.policy.writable_departments(ctx.user.roles))
    if not writable:
        st.info("Your role cannot upload documents. Ask a department owner to add a file.")
        return
    st.caption("Uploaded files are searchable from your next question. Accepted: PDF, DOCX, MD, TXT.")
    uploaded = st.file_uploader("Choose a file", type=["pdf", "docx", "md", "txt"])
    department = st.selectbox("Department", writable,
                              help="Who can read this document is decided by this department.")
    replace = st.checkbox("Replace the file if one with this name already exists")
    max_mb = ctx.settings["ingestion"]["max_upload_mb"]
    if st.button("Upload", type="primary", disabled=uploaded is None):
        try:
            result = save_document(ctx.root, ctx.pipeline.policy, ctx.user, department,
                                   uploaded.name, uploaded.getvalue(), replace=replace,
                                   max_bytes=max_mb * MB)
        except DocumentError as error:
            st.error(str(error))
        else:
            st.success(f"Saved '{result.title}' to {result.department}.")
            if result.pii_warnings:
                st.warning("This document contains personal data "
                           f"({', '.join(result.pii_warnings)}). Remove it before sharing widely.")


# ---------- analytics ----------

def page_analytics() -> None:
    ctx: Ctx = st.session_state["ctx"]
    if not ctx.pipeline.policy.can_view_analytics(ctx.user):
        st.error("Analytics is limited to executives.")
        return
    queries = load_jsonl(ctx.root / ctx.settings["monitoring"]["log_path"])
    feedback = ctx.feedback.entries()
    s = summarize(queries, feedback)
    st.subheader("Usage and quality")
    st.caption("Question text is not shown, to protect employees' privacy.")
    cols = st.columns(5)
    cols[0].metric("Questions", s["queries"])
    cols[1].metric("Refusal rate", f"{s['refusal_rate']:.0%}")
    cols[2].metric("Avg latency", f"{s['avg_latency_ms']:.0f} ms")
    cols[3].metric("Cost (USD)", f"${s['total_cost_usd']:.4f}")
    cols[4].metric("Helpful / not", f"{s['feedback_up']} / {s['feedback_down']}")
    st.markdown("#### Questions per day")
    st.bar_chart(pd.DataFrame(s["daily"]).set_index("day"))
    left, right = st.columns(2)
    with left:
        st.markdown("#### Why questions were refused")
        st.dataframe(pd.DataFrame(list(s["refusal_reasons"].items()), columns=["reason", "count"]),
                     hide_index=True, use_container_width=True)
    with right:
        st.markdown("#### What kinds of answers")
        st.dataframe(pd.DataFrame(list(s["answer_kinds"].items()), columns=["kind", "count"]),
                     hide_index=True, use_container_width=True)
