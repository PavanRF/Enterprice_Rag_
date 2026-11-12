import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag.documents import (DocumentError, delete_document, list_documents,  # noqa: E402
                           safe_filename, save_document)
from rag.rbac import load_access_config  # noqa: E402

POLICY, USERS = load_access_config(ROOT / "config/access.yaml")
FINANCE = USERS.get("priya.sharma")
HR = USERS.get("james.okoro")
CEO = USERS.get("amara.osei")


def _project():
    tmp = tempfile.TemporaryDirectory()
    root = Path(tmp.name)
    (root / "data/docs/finance").mkdir(parents=True)
    (root / "data/docs/hr").mkdir(parents=True)
    (root / "data/docs/hr/secret.md").write_text("Salary secret.", encoding="utf-8")
    return tmp, root


class SafeFilenameTests(unittest.TestCase):
    def test_strips_path_traversal(self):
        self.assertEqual(safe_filename("../../etc/passwd.txt"), "passwd.txt")
        self.assertEqual(safe_filename("..\\..\\evil.md"), "evil.md")

    def test_cleans_unsafe_characters(self):
        self.assertEqual(safe_filename("Q3 report (final)!.PDF"), "Q3_report_final.pdf")


class UploadTests(unittest.TestCase):
    def setUp(self):
        self.tmp, self.root = _project()

    def tearDown(self):
        self.tmp.cleanup()

    def test_finance_can_upload_to_finance(self):
        result = save_document(self.root, POLICY, FINANCE, "finance", "budget.md", b"# Budget\n\nTotal is 5.")
        self.assertEqual(result.source, "finance/budget.md")
        self.assertTrue((self.root / "data/docs/finance/budget.md").exists())

    def test_finance_cannot_upload_to_hr(self):
        with self.assertRaises(DocumentError):
            save_document(self.root, POLICY, FINANCE, "hr", "x.md", b"text here")
        self.assertFalse((self.root / "data/docs/hr/x.md").exists())

    def test_executive_can_upload_anywhere(self):
        save_document(self.root, POLICY, CEO, "hr", "policy.md", b"HR text here.")
        self.assertTrue((self.root / "data/docs/hr/policy.md").exists())

    def test_unsupported_type_rejected(self):
        with self.assertRaises(DocumentError):
            save_document(self.root, POLICY, FINANCE, "finance", "virus.exe", b"MZ")

    def test_empty_and_oversized_rejected(self):
        with self.assertRaises(DocumentError):
            save_document(self.root, POLICY, FINANCE, "finance", "a.md", b"")
        with self.assertRaises(DocumentError):
            save_document(self.root, POLICY, FINANCE, "finance", "a.md", b"x" * 100, max_bytes=50)

    def test_whitespace_only_document_rejected(self):
        with self.assertRaises(DocumentError):
            save_document(self.root, POLICY, FINANCE, "finance", "blank.md", b"   \n\n  ")

    def test_existing_file_needs_replace_flag(self):
        save_document(self.root, POLICY, FINANCE, "finance", "a.md", b"first version here")
        with self.assertRaises(DocumentError):
            save_document(self.root, POLICY, FINANCE, "finance", "a.md", b"second version here")
        save_document(self.root, POLICY, FINANCE, "finance", "a.md", b"second version here", replace=True)
        self.assertIn("second", (self.root / "data/docs/finance/a.md").read_text())

    def test_pii_is_reported_but_upload_allowed(self):
        result = save_document(self.root, POLICY, FINANCE, "finance", "contacts.md",
                               b"Reach ann@example.com or 415-555-0182.")
        self.assertIn("email", result.pii_warnings)
        self.assertIn("phone", result.pii_warnings)

    def test_path_traversal_cannot_write_outside_department(self):
        result = save_document(self.root, POLICY, FINANCE, "finance", "../../../escape.md", b"text here")
        self.assertTrue(result.source.startswith("finance/"))
        self.assertFalse((self.root / "escape.md").exists())


class DeleteAndListTests(unittest.TestCase):
    def setUp(self):
        self.tmp, self.root = _project()
        (self.root / "data/docs/finance/q3.md").write_text("---\ntitle: Q3 Results\n---\n# Q3\n\nRevenue.")
        (self.root / "data/docs/hr/secret.md").write_text("Salary secret.", encoding="utf-8")

    def tearDown(self):
        self.tmp.cleanup()

    def test_finance_cannot_see_hr_documents(self):
        sources = [d["source"] for d in list_documents(self.root, POLICY, FINANCE)]
        self.assertIn("finance/q3.md", sources)
        self.assertNotIn("hr/secret.md", sources)

    def test_listing_shows_title_from_front_matter(self):
        titles = {d["title"] for d in list_documents(self.root, POLICY, FINANCE)}
        self.assertIn("Q3 Results", titles)

    def test_delete_allowed_only_in_writable_department(self):
        with self.assertRaises(DocumentError):
            delete_document(self.root, POLICY, FINANCE, "hr/secret.md")
        self.assertTrue((self.root / "data/docs/hr/secret.md").exists())
        delete_document(self.root, POLICY, FINANCE, "finance/q3.md")
        self.assertFalse((self.root / "data/docs/finance/q3.md").exists())

    def test_delete_rejects_traversal(self):
        with self.assertRaises(DocumentError):
            delete_document(self.root, POLICY, FINANCE, "finance/../hr/secret.md")


if __name__ == "__main__":
    unittest.main()
