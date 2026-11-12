import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag.analytics import summarize  # noqa: E402
from rag.factory import build_pipeline, load_branding  # noqa: E402
from rag.feedback import FeedbackLog  # noqa: E402
from rag.llm import ExtractiveLLM, LLMResult  # noqa: E402
from rag.pipeline import is_catalog_question  # noqa: E402


class RecordingLLM(ExtractiveLLM):
    """Extractive answers, but records the questions it was asked and rewrites follow-ups."""

    def __init__(self):
        self.rewrites = []
        self.asked = []

    def rewrite(self, question, history):
        self.rewrites.append((question, len(history)))
        return LLMResult("What is the status of the Q3 Financial Summary?", 0, 0, self.model)

    def complete(self, question, contexts):
        self.asked.append(question)
        return super().complete(question, contexts)


class PipelineFeatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.log = Path(cls._tmp.name) / "q.jsonl"

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def setUp(self):
        self.llm = RecordingLLM()
        self.pipeline, self.users = build_pipeline(ROOT, llm=self.llm, log_path=self.log)

    def test_catalog_question_lists_only_readable_documents(self):
        answer = self.pipeline.answer(self.users.get("priya.sharma"), "What documents are available?")
        self.assertEqual(answer.kind, "catalog")
        self.assertIn("Q3 Financial Summary", answer.text)
        self.assertNotIn("DataForge", answer.text)
        self.assertNotIn("Payroll", answer.text)

    def test_catalog_does_not_call_the_model(self):
        self.pipeline.answer(self.users.get("priya.sharma"), "What documents are available?")
        self.assertEqual(self.llm.asked, [])

    def test_catalog_for_executive_includes_strategy(self):
        answer = self.pipeline.answer(self.users.get("amara.osei"), "list my documents")
        self.assertIn("2027 Strategy Memo", answer.text)

    def test_follow_up_is_rewritten_using_history(self):
        history = [{"role": "user", "content": "Tell me about Q3 revenue"},
                   {"role": "assistant", "content": "Revenue was $12.4 million."}]
        answer = self.pipeline.answer(self.users.get("priya.sharma"), "what is the status now", history)
        self.assertEqual(self.llm.rewrites[-1][1], 2)
        self.assertEqual(answer.rewritten_question, "What is the status of the Q3 Financial Summary?")
        self.assertIn("status", self.llm.asked[-1].lower())

    def test_no_history_means_no_rewrite_call(self):
        self.pipeline.answer(self.users.get("priya.sharma"), "What was Q3 revenue?")
        self.assertEqual(self.llm.rewrites, [])

    def test_history_is_masked_before_it_reaches_the_model(self):
        history = [{"role": "user", "content": "my card is 4111 1111 1111 1111"},
                   {"role": "assistant", "content": "ok"}]
        captured = {}
        original = self.llm.rewrite

        def spy(question, hist):
            captured["history"] = hist
            return original(question, hist)

        self.llm.rewrite = spy
        self.pipeline.answer(self.users.get("priya.sharma"), "what was Q3 revenue", history)
        self.assertNotIn("4111", captured["history"][0]["content"])

    def test_answer_has_id_for_feedback(self):
        a = self.pipeline.answer(self.users.get("priya.sharma"), "What was Q3 revenue?")
        self.assertEqual(len(a.answer_id), 32)

    def test_suggestions_are_drawn_from_readable_titles(self):
        suggestions = self.pipeline.suggestions(self.users.get("james.okoro"))
        self.assertTrue(suggestions)
        self.assertFalse(any("DataForge" in s or "Q3 Financial" in s for s in suggestions))


class CatalogDetectionTests(unittest.TestCase):
    def test_typo_still_detected(self):
        self.assertTrue(is_catalog_question("what are the avilable documents"))

    def test_topic_question_is_not_catalog(self):
        self.assertFalse(is_catalog_question("Which documents mention invoices?"))


class FeedbackAndAnalyticsTests(unittest.TestCase):
    def test_feedback_roundtrip_stores_no_question_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = FeedbackLog(Path(tmp) / "fb.jsonl")
            log.record("priya.sharma", "abc123", helpful=True)
            log.record("priya.sharma", "def456", helpful=False)
            entries = log.entries()
            self.assertEqual(len(entries), 2)
            self.assertNotIn("question", entries[0])

    def test_summary_counts_refusals_and_feedback_without_question_text(self):
        queries = [
            {"ts": "2026-11-12T10:00:00", "refused": False, "latency_ms": 100, "cost_usd": 0.001,
             "prompt_tokens": 50, "completion_tokens": 10, "question": "secret salary question"},
            {"ts": "2026-11-12T11:00:00", "refused": True, "refusal_reason": "out_of_scope",
             "latency_ms": 2, "cost_usd": 0, "prompt_tokens": 0, "completion_tokens": 0},
        ]
        feedback = [{"helpful": True}, {"helpful": False}, {"helpful": True}]
        import datetime as dt
        s = summarize(queries, feedback, days=3, today=dt.date(2026, 11, 12))
        self.assertEqual(s["queries"], 2)
        self.assertEqual(s["refusals"], 1)
        self.assertEqual(s["refusal_reasons"], {"out_of_scope": 1})
        self.assertEqual(s["feedback_up"], 2)
        self.assertEqual(s["daily"][-1]["queries"], 2)
        self.assertNotIn("secret salary question", str(s))


class BrandingTests(unittest.TestCase):
    def test_branding_has_backgrounds_and_default_in_list(self):
        b = load_branding(ROOT)
        self.assertIn(b["default_background"], b["backgrounds"])
        self.assertGreaterEqual(len(b["backgrounds"]), 2)


if __name__ == "__main__":
    unittest.main()
