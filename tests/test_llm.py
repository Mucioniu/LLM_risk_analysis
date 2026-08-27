import os
import unittest
from unittest.mock import Mock, patch

from credit_assistant.llm import (
    env_reasoning_setting,
    is_ollama_endpoint,
    optional_llm_summary,
)


class LlmClientTests(unittest.TestCase):
    def test_detects_default_ollama_endpoint(self) -> None:
        self.assertTrue(is_ollama_endpoint("http://localhost:11434/v1"))
        self.assertTrue(is_ollama_endpoint("http://127.0.0.1:11434"))
        self.assertFalse(is_ollama_endpoint("https://example.openai.azure.com"))

    def test_parses_explicit_reasoning_settings(self) -> None:
        with patch.dict(os.environ, {"OLLAMA_THINK": "false"}, clear=True):
            self.assertIs(env_reasoning_setting(), False)
        with patch.dict(os.environ, {"OLLAMA_THINK": "high"}, clear=True):
            self.assertEqual(env_reasoning_setting(), "high")
        with patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(env_reasoning_setting())

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
        self.assertIs(payload["think"], False)

    @patch("credit_assistant.llm.httpx.post")
    def test_local_request_preserves_high_reasoning_level(self, post: Mock) -> None:
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "message": {"content": '{"value":1}'},
            "done_reason": "stop",
        }
        post.return_value = response

        environment = {
            "OPENAI_API_KEY": "ollama",
            "OPENAI_BASE_URL": "http://localhost:11434/v1",
            "OPENAI_MODEL": "reasoning-model",
            "OLLAMA_THINK": "high",
            "OPENAI_TEMPERATURE": "0.7",
            "OPENAI_SEED": "42",
        }
        with patch.dict(os.environ, environment, clear=True):
            optional_llm_summary("system", "user", response_format_json=True)

        payload = post.call_args.kwargs["json"]
        self.assertEqual(payload["think"], "high")
        self.assertEqual(payload["options"]["temperature"], 0.7)
        self.assertEqual(payload["options"]["seed"], 42)

    @patch("credit_assistant.llm.httpx.post")
    def test_stage_route_and_exact_json_schema_override_global_settings(self, post: Mock) -> None:
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "message": {"content": '{"value":1}'},
            "done_reason": "stop",
        }
        post.return_value = response
        schema = {
            "type": "object",
            "properties": {"value": {"type": "number"}},
            "required": ["value"],
            "additionalProperties": False,
        }
        environment = {
            "OPENAI_API_KEY": "ollama",
            "OPENAI_BASE_URL": "http://localhost:11434/v1",
            "OPENAI_MODEL": "global-model",
            "OPENAI_CALCULATION_MODEL": "numeric-model",
            "OLLAMA_THINK": "false",
            "OLLAMA_CALCULATION_THINK": "high",
            "OPENAI_TEMPERATURE": "0.1",
            "OPENAI_CALCULATION_TEMPERATURE": "0.6",
            "OLLAMA_NUM_CTX": "8192",
            "OLLAMA_NUM_PREDICT": "3000",
            "OLLAMA_CALCULATION_NUM_CTX": "4096",
            "OLLAMA_CALCULATION_NUM_PREDICT": "6000",
        }

        with patch.dict(os.environ, environment, clear=True):
            optional_llm_summary(
                "system",
                "user",
                response_format_json=True,
                json_schema=schema,
                model_env_name="OPENAI_CALCULATION_MODEL",
                reasoning_env_name="OLLAMA_CALCULATION_THINK",
                temperature_env_name="OPENAI_CALCULATION_TEMPERATURE",
                num_ctx_env_name="OLLAMA_CALCULATION_NUM_CTX",
                num_predict_env_name="OLLAMA_CALCULATION_NUM_PREDICT",
            )

        payload = post.call_args.kwargs["json"]
        self.assertEqual(payload["model"], "numeric-model")
        self.assertEqual(payload["format"], schema)
        self.assertEqual(payload["think"], "high")
        self.assertEqual(payload["options"]["temperature"], 0.6)
        self.assertEqual(payload["options"]["num_ctx"], 4096)
        self.assertEqual(payload["options"]["num_predict"], 6000)


if __name__ == "__main__":
    unittest.main()
