import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag.ingest import ingest_directory  # noqa: E402
from rag.rbac import ALL, AccessPolicy, User, load_access_config  # noqa: E402
from rag.store import TfidfStore  # noqa: E402

DEPTS = ["finance", "hr", "marketing", "executive", "general"]


class PolicyTests(unittest.TestCase):
    def setUp(self):
        self.policy, self.users = load_access_config(ROOT / "config/access.yaml")

    def test_finance_cannot_read_hr(self):
        self.assertNotIn("hr", self.policy.allowed_departments(["finance"]))

    def test_finance_reads_marketing_spend(self):
        self.assertIn("marketing", self.policy.allowed_departments(["finance"]))

    def test_executive_reads_everything(self):
        self.assertEqual(self.policy.allowed_departments(["executive"]), frozenset(DEPTS))

    def test_unknown_role_grants_nothing(self):
        self.assertEqual(self.policy.allowed_departments(["intern"]), frozenset())

    def test_no_roles_grants_nothing(self):
        self.assertEqual(self.policy.allowed_departments([]), frozenset())

    def test_policy_rejects_grant_to_unknown_department(self):
        with self.assertRaises(ValueError):
            AccessPolicy({"x": ["nonexistent"]}, DEPTS)

    def test_unknown_user_raises(self):
        with self.assertRaises(KeyError):
            self.users.get("nobody")


class IngestionTests(unittest.TestCase):
    def test_untagged_document_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "loose.md").write_text("# No department anywhere\n\nSecret text.")
            with self.assertRaises(ValueError):
                ingest_directory(root, DEPTS)

    def test_unknown_folder_department_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "legal").mkdir()
            (root / "legal" / "memo.md").write_text("Memo text.")
            with self.assertRaises(ValueError):
                ingest_directory(root, DEPTS)

    def test_front_matter_department_is_used(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "memo.md").write_text("---\ndepartment: hr\ntitle: Memo\n---\n# Memo\n\nText.")
            chunks = ingest_directory(root, DEPTS)
            self.assertEqual(chunks[0].department, "hr")
            self.assertEqual(chunks[0].title, "Memo")


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.policy, _ = load_access_config(ROOT / "config/access.yaml")
        self.store = TfidfStore()
        self.store.add(ingest_directory(ROOT / "data/docs", DEPTS))

    def test_search_never_returns_disallowed_departments(self):
        allowed = self.policy.allowed_departments(["hr"])
        for query in ["payroll salary band", "DataForge acquisition", "Q3 revenue", "marketing spend"]:
            for hit in self.store.search(query, allowed, k=10):
                self.assertIn(hit.chunk.department, allowed, query)

    def test_restricted_document_is_invisible_to_lower_role(self):
        hits = self.store.search("DataForge acquisition $38 million", frozenset({"finance"}), k=10)
        self.assertFalse(any("DataForge" in h.chunk.text for h in hits))

    def test_empty_allowed_set_returns_nothing(self):
        self.assertEqual(self.store.search("revenue", frozenset(), k=5), [])


if __name__ == "__main__":
    unittest.main()
