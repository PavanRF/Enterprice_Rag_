import datetime as dt
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag.presentation import (background_css, background_options, export_markdown,  # noqa: E402
                              theme_css)
from rag.factory import load_branding  # noqa: E402


class ThemeTests(unittest.TestCase):
    def setUp(self):
        self.branding = load_branding(ROOT)

    def test_preset_background_is_used(self):
        css = background_css(self.branding, "Ocean", ROOT)
        self.assertEqual(css, self.branding["backgrounds"]["Ocean"])

    def test_unknown_choice_falls_back_to_first_preset(self):
        css = background_css(self.branding, "does-not-exist", ROOT)
        self.assertEqual(css, next(iter(self.branding["backgrounds"].values())))

    def test_theme_applies_accent_and_background(self):
        css = theme_css(self.branding, "Sunset", ROOT)
        self.assertIn(self.branding["accent_color"], css)
        self.assertIn(self.branding["backgrounds"]["Sunset"], css)
        self.assertNotIn("__BG__", css)
        self.assertNotIn("__ACCENT__", css)

    def test_photo_option_only_when_image_exists(self):
        self.assertNotIn("Photo", background_options(self.branding, ROOT))
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "assets").mkdir()
            (Path(tmp) / "assets" / "bg.jpg").write_bytes(b"\xff\xd8\xff fake jpeg bytes")
            branding = dict(self.branding, background_image="bg.jpg")
            self.assertIn("Photo", background_options(branding, Path(tmp)))
            css = background_css(branding, "Photo", Path(tmp))
            self.assertIn("data:image/jpeg;base64,", css)

    def test_missing_photo_is_not_offered(self):
        with tempfile.TemporaryDirectory() as tmp:
            branding = dict(self.branding, background_image="missing.jpg")
            self.assertNotIn("Photo", background_options(branding, Path(tmp)))


class ExportTests(unittest.TestCase):
    def test_export_contains_messages_and_sources(self):
        messages = [
            {"role": "user", "content": "What is the meal limit?"},
            {"role": "assistant", "content": "Meals are capped at $75 [1].",
             "citations": [{"title": "Expense Policy", "source": "finance/expense_policy.md"}]},
        ]
        text = export_markdown(messages, "Priya Sharma", now=dt.datetime(2026, 11, 12, 9, 30))
        self.assertIn("Chat export: Priya Sharma", text)
        self.assertIn("**You:** What is the meal limit?", text)
        self.assertIn("Source: Expense Policy (finance/expense_policy.md)", text)
        self.assertIn("2026-11-12 09:30", text)


if __name__ == "__main__":
    unittest.main()
