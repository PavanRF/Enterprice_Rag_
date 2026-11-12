"""Pure presentation helpers (no Streamlit import), so they can be unit tested."""
from __future__ import annotations

import base64
import datetime as dt
import mimetypes
from pathlib import Path

PHOTO_OPTION = "Photo"

_THEME_CSS = """
<style>
.stApp { background: __BG__; }
.block-container {
  background: rgba(15, 23, 42, 0.74); border-radius: 18px;
  padding: 2rem 2.5rem; backdrop-filter: blur(10px);
  animation: fade-up 0.6s ease both;
}
[data-testid="stSidebar"] > div:first-child { background: rgba(2, 6, 23, 0.86); }
h1, h2, h3, p, label, li { color: #f8fafc !important; }
button[kind="primary"], button[kind="primaryFormSubmit"] {
  background: __ACCENT__ !important; border-color: __ACCENT__ !important; color: #ffffff !important;
}
[data-testid="stChatMessage"] { animation: fade-up 0.35s ease both; border-radius: 12px; }
.hero { text-align: center; margin: 0.5rem 0 1.5rem; }
.hero h1 { font-size: 2.4rem; margin-bottom: 0.25rem; }
.hero p { opacity: 0.85; font-size: 1.05rem; }
@keyframes fade-up {
  from { opacity: 0; transform: translateY(10px); }
  to { opacity: 1; transform: none; }
}
</style>
"""


def background_options(branding: dict, root: Path) -> list[str]:
    options = list(branding.get("backgrounds", {}))
    if photo_path(branding, root):
        options.append(PHOTO_OPTION)
    return options


def photo_path(branding: dict, root: Path) -> Path | None:
    name = branding.get("background_image") or ""
    if not name:
        return None
    path = Path(root) / "assets" / name
    return path if path.is_file() else None


def background_css(branding: dict, choice: str, root: Path) -> str:
    if choice == PHOTO_OPTION:
        path = photo_path(branding, root)
        if path:
            mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
            data = base64.b64encode(path.read_bytes()).decode("ascii")
            return (f'linear-gradient(rgba(2, 6, 23, 0.55), rgba(2, 6, 23, 0.55)), '
                    f'url("data:{mime};base64,{data}") center / cover fixed')
    presets = branding.get("backgrounds", {})
    if choice in presets:
        return presets[choice]
    return next(iter(presets.values()), "#0f172a")


def theme_css(branding: dict, choice: str, root: Path) -> str:
    accent = branding.get("accent_color", "#4F8BF9")
    return (_THEME_CSS
            .replace("__BG__", background_css(branding, choice, root))
            .replace("__ACCENT__", accent))


def export_markdown(messages: list[dict], name: str, now: dt.datetime | None = None) -> str:
    now = now or dt.datetime.now()
    lines = [f"# Chat export: {name}", f"_Exported {now:%Y-%m-%d %H:%M}_", ""]
    for m in messages:
        who = "You" if m["role"] == "user" else "Assistant"
        lines += [f"**{who}:** {m['content']}", ""]
        for c in m.get("citations", []):
            lines.append(f"- Source: {c['title']} ({c['source']})")
        if m.get("citations"):
            lines.append("")
    return "\n".join(lines)


def citation_label(c: dict) -> str:
    return f"[{c['n']}] {c['title']} · {c['department']} · {c['source']}"
