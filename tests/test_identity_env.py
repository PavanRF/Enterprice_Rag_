import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag.env import load_dotenv  # noqa: E402
from rag.factory import source_signature  # noqa: E402
from rag.identity import AuthError, user_from_header  # noqa: E402
from rag.rbac import load_access_config  # noqa: E402


class EnvTests(unittest.TestCase):
    def test_loads_values_skips_comments_and_inline_comments(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / ".env"
            path.write_text("# comment\nRAG_TEST_A=hello\nRAG_TEST_B=world # note\n", encoding="utf-8")
            try:
                load_dotenv(path)
                self.assertEqual(os.environ["RAG_TEST_A"], "hello")
                self.assertEqual(os.environ["RAG_TEST_B"], "world")
            finally:
                os.environ.pop("RAG_TEST_A", None)
                os.environ.pop("RAG_TEST_B", None)

    def test_does_not_override_existing_environment(self):
        os.environ["RAG_TEST_C"] = "from-platform"
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / ".env"
            path.write_text("RAG_TEST_C=from-file\n", encoding="utf-8")
            try:
                load_dotenv(path)
                self.assertEqual(os.environ["RAG_TEST_C"], "from-platform")
            finally:
                os.environ.pop("RAG_TEST_C", None)


class IdentityTests(unittest.TestCase):
    def setUp(self):
        _, self.users = load_access_config(ROOT / "config/access.yaml")

    def test_known_identity_resolves(self):
        self.assertEqual(user_from_header(self.users, "priya.sharma").user_id, "priya.sharma")

    def test_missing_identity_is_denied(self):
        for value in [None, "", "   "]:
            with self.assertRaises(AuthError):
                user_from_header(self.users, value)

    def test_unknown_identity_is_denied(self):
        with self.assertRaises(AuthError):
            user_from_header(self.users, "intruder@example.com")


class SignatureTests(unittest.TestCase):
    def test_signature_changes_when_a_document_is_added(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "config").mkdir()
            (root / "config/access.yaml").write_text("departments: []\n", encoding="utf-8")
            (root / "data/docs/finance").mkdir(parents=True)
            (root / "data/docs/finance/a.md").write_text("Revenue.", encoding="utf-8")
            before = source_signature(root)
            self.assertEqual(before, source_signature(root))  # stable when nothing changes
            (root / "data/docs/finance/b.md").write_text("New document.", encoding="utf-8")
            self.assertNotEqual(before, source_signature(root))


if __name__ == "__main__":
    unittest.main()
