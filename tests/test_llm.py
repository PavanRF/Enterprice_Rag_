import io
import json
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag.llm import ExtractiveLLM, OpenAICompatibleLLM, build_user_prompt  # noqa: E402


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class OpenAICompatibleTests(unittest.TestCase):
    def test_sends_grounded_prompt_and_parses_usage(self):
        reply = {
            "choices": [{"message": {"content": "Q3 revenue was $12.4 million [1]."}}],
            "usage": {"prompt_tokens": 120, "completion_tokens": 9},
        }
        captured = {}

        def fake_urlopen(request, timeout):
            captured["url"] = request.full_url
            captured["auth"] = request.get_header("Authorization")
            captured["body"] = json.loads(request.data)
            return FakeResponse(json.dumps(reply).encode())

        client = OpenAICompatibleLLM("test-key", "test-model")
        with mock.patch("urllib.request.urlopen", side_effect=fake_urlopen):
            result = client.complete("What was Q3 revenue?", ["Q3 revenue reached $12.4 million."])

        self.assertEqual(captured["url"], "https://api.groq.com/openai/v1/chat/completions")
        self.assertEqual(captured["auth"], "Bearer test-key")
        self.assertEqual(captured["body"]["temperature"], 0)
        self.assertEqual(captured["body"]["messages"][0]["role"], "system")
        self.assertIn("[1] Q3 revenue reached", captured["body"]["messages"][1]["content"])
        self.assertEqual(result.text, "Q3 revenue was $12.4 million [1].")
        self.assertEqual((result.prompt_tokens, result.completion_tokens), (120, 9))

    def test_prompt_numbers_sources(self):
        prompt = build_user_prompt("Q?", ["alpha", "beta"])
        self.assertIn("[1] alpha", prompt)
        self.assertIn("[2] beta", prompt)


class ExtractiveTests(unittest.TestCase):
    def test_answers_with_supporting_sentence_and_citation(self):
        result = ExtractiveLLM().complete(
            "How many weeks of parental leave?",
            ["Parental leave is 16 weeks of paid leave for the primary caregiver."])
        self.assertTrue(result.answerable)
        self.assertIn("16 weeks", result.text)
        self.assertIn("[1]", result.text)

    def test_declines_when_no_sentence_is_supported(self):
        result = ExtractiveLLM().complete("What is the dragon policy?",
                                          ["Passwords must be at least 14 characters."])
        self.assertFalse(result.answerable)

    def test_headings_are_not_quoted(self):
        result = ExtractiveLLM().complete("What is the IT security policy?",
                                          ["# IT Security Policy\n\nMFA is required for systems."])
        self.assertFalse(result.answerable)


if __name__ == "__main__":
    unittest.main()
