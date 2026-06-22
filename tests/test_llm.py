import os
import unittest
from unittest.mock import Mock, patch

from credit_assistant.llm import is_ollama_endpoint, optional_llm_summary


class LlmClientTests(unittest.TestCase):
    def test_detects_default_ollama_endpoint(self) -> None:
        self.assertTrue(is_ollama_endpoint("http://localhost:11434/v1"))
        self.assertTrue(is_ollama_endpoint("http://127.0.0.1:11434"))
        self.assertFalse(is_ollama_endpoint("https://example.openai.azure.com"))

    @patch("credit_assistant.llm.httpx.post")
    def test_local_json_request_uses_native_ollama_with_larger_context(self, post: Mock) -> None:
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "message": {"content": '{"value":"**kept verbatim**"}'},
            "done_reason": "stop",
        }
        post.return_value = response

        environment = {
            "OPENAI_API_KEY": "ollama",
            "OPENAI_BASE_URL": "http://localhost:11434/v1",
            "OPENAI_MODEL": "mistral-small3.2",
            "OLLAMA_THINK": "false",
        }
        with patch.dict(os.environ, environment, clear=True):
            result = optional_llm_summary(
                "system",
                "user",
                response_format_json=True,
                max_tokens_override=3000,
            )

        self.assertEqual(result, '{"value":"**kept verbatim**"}')
        request_url = post.call_args.args[0]
        payload = post.call_args.kwargs["json"]
        self.assertEqual(request_url, "http://localhost:11434/api/chat")
        self.assertEqual(payload["format"], "json")
        self.assertEqual(payload["options"]["num_ctx"], 8192)
        self.assertEqual(payload["options"]["num_predict"], 3000)
        self.assertNotIn("think", payload)


if __name__ == "__main__":
    unittest.main()
