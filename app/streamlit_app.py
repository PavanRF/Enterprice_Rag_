"""Entry point: streamlit run app/streamlit_app.py

AUTH_MODE=login (default): users sign in with a username and password.
AUTH_MODE=google: users sign in with Google; only emails listed in access.yaml get in.
AUTH_MODE=proxy: identity comes from the SSO layer via the AUTH_HEADER header.
"""
import os
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "app"))

st.set_page_config(page_title="Northwind Assistant", page_icon="🔐", layout="wide")

from rag.auth import AuthError, LoginLimiter  # noqa: E402
from rag.env import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from rag.factory import (build_pipeline, credentials_path, llm_mode,  # noqa: E402
                         load_branding, load_settings, source_signature)
from rag.feedback import FeedbackLog  # noqa: E402
from rag.identity import user_from_google, user_from_header  # noqa: E402
from rag.rbac import UnknownUserError  # noqa: E402
import ui  # noqa: E402

AUTH_MODE = os.environ.get("AUTH_MODE", "login").lower()
AUTH_HEADER = os.environ.get("AUTH_HEADER", "X-Forwarded-User")


@st.cache_resource
def get_app(signature: str):
    # Editing or adding a document changes the signature, so the index rebuilds
    # automatically on the next interaction.
    return build_pipeline(ROOT)


@st.cache_resource
def get_limiter():
    auth = load_settings(ROOT)["auth"]
    return LoginLimiter(max_failures=auth["max_failed_logins"], lockout_seconds=auth["lockout_seconds"])


branding = load_branding(ROOT)
settings = load_settings(ROOT)
pipeline, users = get_app(source_signature(ROOT))
bg_choice = st.session_state.get("bg", branding.get("default_background"))
ui.apply_theme(branding, bg_choice, ROOT)

if AUTH_MODE == "proxy":
    try:
        user = user_from_header(users, st.context.headers.get(AUTH_HEADER))
    except AuthError as error:
        st.error(f"Access denied: {error}")
        st.stop()
elif AUTH_MODE == "google":
    # st.login / st.user need [auth] settings in .streamlit/secrets.toml (see README).
    try:
        google_email = dict(st.user).get("email")
    except Exception:
        google_email = None
    if not google_email:
        ui.render_google_login(branding)
        st.stop()
    try:
        user = user_from_google(users, google_email)
    except AuthError as error:
        st.error(str(error))
        if st.button("Sign in with a different account"):
            st.logout()
        st.stop()
else:
    user = None
    user_id = st.session_state.get("user_id")
    if user_id:
        try:
            user = users.get(user_id)
        except UnknownUserError:
            st.session_state.clear()
    if user is None:
        ui.render_login(branding, users, credentials_path(ROOT), get_limiter(), ROOT)
        st.stop()

ctx = ui.Ctx(
    root=ROOT, pipeline=pipeline, users=users, user=user, branding=branding, settings=settings,
    feedback=FeedbackLog(ROOT / settings["monitoring"]["feedback_path"]),
    creds_file=credentials_path(ROOT), limiter=get_limiter(), llm=llm_mode(),
    auth_mode=AUTH_MODE,
)
st.session_state["ctx"] = ctx

pages = [
    st.Page(ui.page_assistant, title="Assistant", icon="💬", default=True),
    st.Page(ui.page_documents, title="Documents", icon="📚"),
    st.Page(ui.page_upload, title="Upload", icon="⬆️"),
]
if pipeline.policy.can_view_analytics(user):
    pages.append(st.Page(ui.page_analytics, title="Analytics", icon="📊"))

navigation = st.navigation(pages)
with st.sidebar:
    ui.render_sidebar(ctx)
navigation.run()
