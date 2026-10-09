"""审核问题回归：只使用假凭据、临时目录和 mock。"""
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
from urllib.parse import quote

from test_generate_image import generate_image as g, PNG_BYTES, parse_args
from image_gen.errors import error_payload


class CLIAuditFixes(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.keys = {"OPENAI_API_KEY": "audit-openai-key+/", "GEMINI_API_KEY": "audit-gemini-key+/"}
        env = patch.dict(os.environ, {**self.keys, "OPENAI_API_BASE_URL": "https://openai.example",
                                      "GEMINI_API_BASE_URL": "https://gemini.example"}, clear=True)
        env.start()
        self.addCleanup(env.stop)
        loader = patch.object(g, "load_environment", return_value=[])
        loader.start()
        self.addCleanup(loader.stop)
        network = patch.object(g.requests.sessions.Session, "request", side_effect=AssertionError("Network forbidden"))
        network.start()
        self.addCleanup(network.stop)

    def main_error(self, *arguments):
        stderr, stdout = io.StringIO(), io.StringIO()
        with patch.object(sys, "argv", [g.__file__, *arguments]), patch.object(sys, "stderr", stderr), \
             patch.object(sys, "stdout", stdout):
            code = g.main()
        self.assertEqual(stdout.getvalue(), "")
        return code, json.loads(stderr.getvalue())

    def test_http_error_redacts_both_configured_keys_and_keeps_diagnostics(self):
        response = Mock(status_code=401, headers={})
        response.json.return_value = {"error": {"message": "invalid " + " ".join(self.keys.values())}}
        with patch.object(g.requests, "post", return_value=response):
            code, payload = self.main_error("generate", "--prompt", "audit", "--output-dir", str(self.root))
        self.assertEqual(code, 1)
        self.assertEqual(payload["http_status"], 401)
        self.assertEqual(payload["error_code"], "api_http_error")
        for key in self.keys.values():
            self.assertNotIn(key, json.dumps(payload))

    def test_redacts_encoded_keys_and_unknown_auth_or_url_secrets(self):
        message = (quote(self.keys["OPENAI_API_KEY"], safe="") + " "
                   "Authorization: Bearer unknown-bearer\nx-goog-api-key=unknown-header\n"
                   "https://user:password@cdn.example/img?token=unknown-query&width=1024")
        result = error_payload(ValueError(message))["error"]
        for secret in ("unknown-bearer", "unknown-header", "unknown-query", "user:password", "audit-openai"):
            self.assertNotIn(secret, result)
        self.assertIn("width=1024", result)

    def test_argument_failures_are_json_and_do_not_load_configuration(self):
        for arguments in (("generate", "--prompt", "test", "--image-size", "3K"),
                          ("generate", "--unknown", "value"), (), ("recover",)):
            with self.subTest(arguments=arguments), patch.object(g, "load_environment") as loader:
                code, payload = self.main_error(*arguments)
                self.assertEqual(code, 2)
                self.assertEqual(payload["error_code"], "argument_error")
                self.assertEqual(payload["stage"], "arguments")
                loader.assert_not_called()

    def test_unknown_basic_digest_and_quoted_authorization_are_fully_redacted(self):
        for header in ("Authorization: Basic ZmFrZTpzZWNyZXQ=",
                       'Authorization: Digest username="private-user", nonce="private-nonce", response="private-response"',
                       "{'Authorization': 'Basic ZmFrZTpzZWNyZXQ='}"):
            with self.subTest(header=header):
                output = error_payload(ValueError(header + "\nHTTP 401 from gateway"))["error"]
                for secret in ("ZmFrZTpzZWNyZXQ=", "private-user", "private-nonce", "private-response"):
                    self.assertNotIn(secret, output)
                self.assertIn("HTTP 401 from gateway", output)

    def test_help_keeps_normal_successful_text_output(self):
        stdout = io.StringIO()
        with patch.object(sys, "argv", [g.__file__, "--help"]), patch.object(sys, "stdout", stdout):
            with self.assertRaises(SystemExit) as result:
                g.main()
        self.assertEqual(result.exception.code, 0)
        self.assertIn("usage:", stdout.getvalue())

    def test_doctor_rejects_incompatible_default_format_or_quality(self):
        for configuration, field in (({"OPENAI_SUPPORTED_FORMATS": "jpeg"}, "DEFAULT_OUTPUT_FORMAT"),
                                     ({"OPENAI_SUPPORTED_QUALITIES": "high"}, "DEFAULT_QUALITY"),
                                     ({"GEMINI_DEFAULT_OUTPUT_FORMAT": "jpeg"}, "DEFAULT_OUTPUT_FORMAT"),
                                     ({"GEMINI_DEFAULT_QUALITY": "high", "GEMINI_SUPPORTED_QUALITIES": "high"}, "Gemini")):
            api = "gemini" if any(k.startswith("GEMINI") for k in configuration) else "openai"
            with self.subTest(configuration=configuration), patch.dict(os.environ, configuration):
                with self.assertRaisesRegex(ValueError, field):
                    g.run(parse_args("doctor", "--api-format", api))

    def test_configured_default_format_quality_match_doctor_and_request(self):
        with patch.dict(os.environ, {"OPENAI_SUPPORTED_FORMATS": "jpeg", "OPENAI_DEFAULT_OUTPUT_FORMAT": "jpeg",
                                     "OPENAI_SUPPORTED_QUALITIES": "high", "OPENAI_DEFAULT_QUALITY": "high"}):
            doctor = g.run(parse_args("doctor"))
            plan = g.run(parse_args("generate", "--prompt", "audit", "--dry-run", "--output-dir", str(self.root)))
        self.assertEqual(doctor["status"], "ok")
        self.assertEqual(doctor["capabilities"]["default_output_format"], "jpeg")
        self.assertEqual(plan["parameters"]["output_format"], "jpeg")
        self.assertEqual(plan["parameters"]["quality"], "high")

    def test_reference_request_preserves_user_layout_requirement(self):
        source = self.root / "reference.png"
        source.write_bytes(PNG_BYTES)
        instruction = "保留参考图的构图和布局，把汽车替换成咖啡杯"
        with patch.object(g.requests, "post") as api:
            plan = g.run(parse_args("reference", "--image", str(source), "--prompt", instruction,
                                    "--output-dir", str(self.root), "--dry-run"))
            api.assert_not_called()
        self.assertIn(instruction, plan["effective_prompt"])
        self.assertNotIn("全新构图", plan["effective_prompt"])

    def test_storage_failure_is_pending_and_has_recovery_path(self):
        with patch.object(g, "call_openai", return_value=(PNG_BYTES, "image/png", "https://openai.example")), \
             patch.object(g, "write_image_exclusive", side_effect=OSError("disk unavailable")):
            code, payload = self.main_error("generate", "--prompt", "audit", "--output-dir", str(self.root))
        self.assertEqual(code, 1)
        self.assertEqual(payload["error_code"], "storage_pending")
        self.assertEqual(payload["stage"], "storage")
        self.assertFalse(payload["retryable"])
        self.assertTrue(Path(payload["recovery_journal"]).is_file())

    def test_failure_to_save_journal_is_storage_error_without_retry(self):
        with patch.object(g, "call_openai", return_value=(PNG_BYTES, "image/png", "https://openai.example")), \
             patch.object(g, "write_json_atomic", side_effect=OSError("disk unavailable")):
            code, payload = self.main_error("generate", "--prompt", "audit", "--output-dir", str(self.root))
        self.assertEqual(code, 1)
        self.assertEqual(payload["error_code"], "storage_failed")
        self.assertEqual(payload["stage"], "storage")
        self.assertFalse(payload["retryable"])


if __name__ == "__main__":
    unittest.main()
