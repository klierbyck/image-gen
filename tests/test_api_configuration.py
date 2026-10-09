"""独立 API 配置与显式路由回归：真实请求构造，HTTP 仅使用 mock。"""
import base64
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

from test_generate_image import generate_image as g, PNG_BYTES, parse_args


class APIConfigurationTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.config = {
            "OPENAI_API_BASE_URL": "https://openai.example/gateway/v1",
            "OPENAI_API_KEY": "openai-fake-key",
            "OPENAI_MODEL": "OpenAIImage-v3",
            "GEMINI_API_BASE_URL": "https://gemini.example/v1beta",
            "GEMINI_API_KEY": "gemini-fake-key",
            "GEMINI_MODEL": "GeminiImage-v3",
        }
        for patcher in (
            patch.dict(os.environ, self.config, clear=True),
            patch.object(g, "load_environment", return_value=[]),
            patch.object(g.requests.sessions.Session, "request", side_effect=AssertionError("Network forbidden")),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    def response(self, api_format):
        data = base64.b64encode(PNG_BYTES).decode("ascii")
        response = Mock(status_code=200)
        response.headers = {"x-request-id": f"request-{api_format}"}
        response.json.return_value = (
            {"data": [{"b64_json": data}], "usage": {"total_tokens": 12}}
            if api_format == "openai"
            else {
                "candidates": [{"content": {"parts": [{"inlineData": {
                    "mimeType": "image/png", "data": data,
                }}]}}],
                "usageMetadata": {"totalTokenCount": 13},
            }
        )
        return response

    def run_image(self, command="generate", *options):
        return g.run(parse_args(command, "--prompt", "API integration test",
                                "--output-dir", str(self.root), *options))

    def test_missing_selected_address_never_uses_other_api(self):
        for api_format in ("openai", "gemini"):
            with self.subTest(api_format=api_format), patch.dict(os.environ, {
                f"{api_format.upper()}_API_BASE_URL": "",
            }), patch.object(g.requests, "post") as post:
                with self.assertRaisesRegex(ValueError, f"{api_format.upper()}_API_BASE_URL"):
                    self.run_image("generate", "--api-format", api_format)
                post.assert_not_called()

    def test_missing_selected_key_never_uses_other_api(self):
        for api_format in ("openai", "gemini"):
            with self.subTest(api_format=api_format), patch.dict(os.environ, {
                f"{api_format.upper()}_API_KEY": "",
            }), patch.object(g.requests, "post") as post:
                with self.assertRaisesRegex(ValueError, f"{api_format.upper()}_API_KEY"):
                    self.run_image("generate", "--api-format", api_format)
                post.assert_not_called()

    def test_address_override_keeps_selected_key_and_model(self):
        with patch.object(g.requests, "post", return_value=self.response("gemini")) as post:
            result = self.run_image("generate", "--api-format", "gemini",
                                    "--base-url", "https://explicit.example/v1beta/")
        self.assertEqual(result["endpoint"], "https://explicit.example/v1beta/models/GeminiImage-v3:generateContent")
        self.assertEqual(post.call_args.kwargs["headers"]["Authorization"], "Bearer gemini-fake-key")

    def test_default_openai_generate_reference_and_edit(self):
        with patch.object(g.requests, "post", return_value=self.response("openai")) as post:
            created = self.run_image()
            self.assertEqual(post.call_args.args[0], "https://openai.example/gateway/v1/images/generations")
            self.assertEqual(post.call_args.kwargs["json"]["model"], "OpenAIImage-v3")
            self.assertEqual(post.call_args.kwargs["headers"]["Authorization"], "Bearer openai-fake-key")
            for command, options in (
                ("reference", ("--image", created["image"])),
                ("edit", ("--session", created["session"])),
            ):
                with self.subTest(command=command):
                    result = self.run_image(command, *options)
                    self.assertEqual(result["api_format"], "openai")
                    self.assertEqual(post.call_args.args[0], "https://openai.example/gateway/v1/images/edits")
                    self.assertEqual(post.call_args.kwargs["data"]["model"], "OpenAIImage-v3")
                    self.assertTrue(post.call_args.kwargs["files"])
                    self.assertFalse(post.call_args.kwargs["allow_redirects"])

    def test_explicit_gemini_generate_reference_and_edit(self):
        with patch.object(g.requests, "post", return_value=self.response("gemini")) as post:
            created = self.run_image("generate", "--api-format", "gemini")
            self.assertEqual(post.call_args.args[0], "https://gemini.example/v1beta/models/GeminiImage-v3:generateContent")
            self.assertEqual(post.call_args.kwargs["headers"]["Authorization"], "Bearer gemini-fake-key")
            for command, options in (
                ("reference", ("--image", created["image"])),
                ("edit", ("--session", created["session"])),
            ):
                self.run_image(command, "--api-format", "gemini", *options)
                parts = post.call_args.kwargs["json"]["contents"][0]["parts"]
                self.assertEqual(base64.b64decode(parts[1]["inlineData"]["data"]), PNG_BYTES)
                self.assertFalse(post.call_args.kwargs["allow_redirects"])

    def test_model_names_and_old_environment_cannot_select_gemini(self):
        # 旧的共用配置不再参与选择，模型名也不影响 API 路由。
        with patch.dict(os.environ, {"IMAGE_API_FORMAT": "gemini", "IMAGE_MODEL": "LegacyModel",
                                    "IMAGE_API_BASE_URL": "https://legacy.example",
                                    "IMAGE_API_KEY": "legacy-key"}):
            for model in ("gemini-custom", "nana-banana-2", "banana", "vendor/MyImage-v3"):
                with self.subTest(model=model):
                    plan = self.run_image("generate", "--model", model, "--dry-run")
                    self.assertEqual(plan["api_format"], "openai")
                    self.assertEqual(plan["model"], model)
                    self.assertEqual(plan["endpoint"], "https://openai.example/gateway/v1/images/generations")
            self.assertEqual(self.run_image("generate", "--dry-run")["model"], "OpenAIImage-v3")

    def test_each_api_has_its_own_default_model(self):
        with patch.dict(os.environ, {"OPENAI_MODEL": "", "GEMINI_MODEL": ""}):
            self.assertEqual(self.run_image("generate", "--dry-run")["model"], "gpt-image-2")
            self.assertEqual(self.run_image("generate", "--api-format", "gemini", "--dry-run")["model"], "nana-banana-2")

    def test_auth_configuration_is_independent(self):
        with patch.dict(os.environ, {"GEMINI_API_AUTH": "x-goog-api-key"}):
            self.assertEqual(g.api_headers("openai-key", "openai"), {"Authorization": "Bearer openai-key"})
            with patch.object(g.requests, "post", return_value=self.response("gemini")) as post:
                self.run_image("generate", "--api-format", "gemini")
                self.assertEqual(post.call_args.kwargs["headers"], {
                    "x-goog-api-key": "gemini-fake-key", "Content-Type": "application/json",
                })

    def test_edit_rejects_implicit_switch_after_gemini_session(self):
        with patch.object(g.requests, "post", return_value=self.response("gemini")):
            created = self.run_image("generate", "--api-format", "gemini", "--model", "SavedGeminiModel")
        with self.assertRaisesRegex(ValueError, "--allow-api-switch"):
            self.run_image("edit", "--session", created["session"], "--dry-run")
        default_plan = self.run_image(
            "edit", "--session", created["session"], "--allow-api-switch", "--dry-run"
        )
        self.assertEqual(default_plan["api_format"], "openai")
        self.assertEqual(default_plan["model"], "OpenAIImage-v3")
        self.assertEqual(default_plan["endpoint"], "https://openai.example/gateway/v1/images/edits")
        with patch.dict(os.environ, {"GEMINI_API_BASE_URL": "https://next.example/v1beta"}):
            plan = self.run_image("edit", "--session", created["session"], "--api-format", "gemini", "--dry-run")
            self.assertEqual(plan["model"], "SavedGeminiModel")
            self.assertEqual(plan["endpoint"], "https://next.example/v1beta/models/SavedGeminiModel:generateContent")
            override = self.run_image("edit", "--session", created["session"], "--api-format", "gemini",
                                      "--model", "NewGeminiModel", "--dry-run")
            self.assertEqual(override["model"], "NewGeminiModel")

    def test_switching_openai_session_to_gemini_requires_opt_in(self):
        with patch.object(g.requests, "post", return_value=self.response("openai")):
            created = self.run_image()
        with self.assertRaisesRegex(ValueError, "--allow-api-switch"):
            self.run_image(
                "edit", "--session", created["session"], "--api-format", "gemini", "--dry-run"
            )
        plan = self.run_image(
            "edit", "--session", created["session"], "--api-format", "gemini",
            "--allow-api-switch", "--dry-run"
        )
        self.assertEqual(plan["model"], "GeminiImage-v3")
        self.assertEqual(plan["endpoint"], "https://gemini.example/v1beta/models/GeminiImage-v3:generateContent")

    def test_doctor_checks_each_configuration_without_exposing_key(self):
        for api_format, expected_model in (
            ("openai", "OpenAIImage-v3"),
            ("gemini", "GeminiImage-v3"),
        ):
            with self.subTest(api_format=api_format):
                result = g.run(parse_args("doctor", "--api-format", api_format))
                self.assertEqual(result["status"], "ok")
                self.assertEqual(result["model"], expected_model)
                self.assertTrue(result["key_configured"])
                self.assertNotIn(self.config[f"{api_format.upper()}_API_KEY"], json.dumps(result))

    def test_doctor_fails_when_selected_key_is_missing(self):
        with patch.dict(os.environ, {"GEMINI_API_KEY": ""}):
            with self.assertRaisesRegex(ValueError, "GEMINI_API_KEY"):
                g.run(parse_args("doctor", "--api-format", "gemini"))

    def test_capability_configuration_changes_defaults_and_limits(self):
        with patch.dict(os.environ, {
            "OPENAI_DEFAULT_SIZE": "1024x1024",
            "OPENAI_SUPPORTED_FORMATS": "png,webp",
            "OPENAI_SUPPORTED_QUALITIES": "auto,high",
            "OPENAI_MAX_INPUT_IMAGE_MB": "1",
        }):
            plan = self.run_image("generate", "--dry-run")
            self.assertEqual(plan["parameters"]["size"], "1024x1024")
            self.assertEqual(plan["capabilities"]["max_input_image_bytes"], 1024 * 1024)
            with self.assertRaisesRegex(ValueError, "不支持输出格式"):
                self.run_image("generate", "--output-format", "jpeg", "--dry-run")

            oversized = self.root / "oversized.png"
            oversized.write_bytes(b"x" * (1024 * 1024 + 1))
            with self.assertRaisesRegex(ValueError, "1 MB"):
                self.run_image("reference", "--image", str(oversized), "--dry-run")

    def test_http_failure_has_structured_retry_metadata(self):
        response = Mock(status_code=503, headers={})
        response.json.return_value = {"error": {"message": "temporarily busy"}}
        response.text = "temporarily busy"
        stderr = io.StringIO()
        argv = [
            g.__file__, "generate", "--prompt", "test", "--output-dir", str(self.root)
        ]
        with patch.object(g.requests, "post", return_value=response), \
             patch.object(sys, "argv", argv), patch.object(sys, "stderr", stderr):
            self.assertEqual(g.main(), 1)
        payload = json.loads(stderr.getvalue())
        self.assertEqual(payload["error_code"], "api_http_error")
        self.assertEqual(payload["stage"], "api")
        self.assertEqual(payload["provider"], "openai")
        self.assertTrue(payload["retryable"])
        self.assertEqual(payload["http_status"], 503)

    def test_api_format_saved_in_manifest_and_session(self):
        manifest = self.root / "manifest.json"
        with patch.object(g.requests, "post", return_value=self.response("openai")):
            created = self.run_image("generate", "--manifest", str(manifest), "--asset-id", "image-01")
        self.assertEqual(json.loads(manifest.read_text())["assets"]["image-01"]["api_format"], "openai")
        self.assertEqual(json.loads(Path(created["session"]).read_text())["api_format"], "openai")

    def test_request_metadata_is_returned_and_persisted(self):
        manifest = self.root / "metadata.json"
        with patch.object(g.requests, "post", return_value=self.response("openai")):
            created = self.run_image(
                "generate", "--manifest", str(manifest), "--asset-id", "image-01"
            )
        self.assertEqual(created["http_status"], 200)
        self.assertEqual(created["request_id"], "request-openai")
        self.assertEqual(created["usage"], {"total_tokens": 12})
        asset = json.loads(manifest.read_text())["assets"]["image-01"]
        turn = json.loads(Path(created["session"]).read_text())["turns"][-1]
        for record in (asset, turn):
            self.assertEqual(record["request_id"], "request-openai")
            self.assertEqual(record["http_status"], 200)
            self.assertIn("elapsed_seconds", record)

    def test_auto_protocol_and_invalid_auth_rejected(self):
        with self.assertRaises(ValueError):
            g.resolve_api_format("auto")
        for api_format in ("openai", "gemini"):
            with self.subTest(api_format=api_format), patch.dict(os.environ, {
                f"{api_format.upper()}_API_AUTH": "unsupported",
            }), self.assertRaises(ValueError):
                self.run_image("generate", "--api-format", api_format, "--dry-run")

    def test_model_path_is_encoded_and_invalid_ids_rejected(self):
        self.assertEqual(g.api_endpoint("https://api.example/v1", "gemini", "vendor/Model", False),
                         "https://api.example/v1/models/vendor%2FModel:generateContent")
        for model in ("bad?key=value", "bad#fragment", "bad\nname", " "):
            with self.subTest(model=model), self.assertRaises(ValueError):
                g.normalize_model(model)

    def test_custom_header_download_never_follows_redirect(self):
        response = Mock(status_code=302, text="redirect")
        with patch.dict(os.environ, {"OPENAI_API_AUTH": "x-goog-api-key"}), \
                patch.object(g.requests, "get", return_value=response) as get:
            with self.assertRaises(g.ImageAPIError):
                g.download_image("https://api.example/image.png", "https://api.example", "fake", 10)
            self.assertFalse(get.call_args.kwargs["allow_redirects"])
            self.assertEqual(get.call_args.kwargs["headers"], {"x-goog-api-key": "fake"})


class EnvironmentFileTests(unittest.TestCase):
    def test_environment_file_loads_both_configurations(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "config.env"
            config.write_text("OPENAI_API_BASE_URL=https://openai.example\nOPENAI_MODEL=CustomOpenAI\n"
                              "GEMINI_API_BASE_URL=https://gemini.example\nGEMINI_MODEL=CustomGemini\n", encoding="utf-8")
            with patch.dict(os.environ, {"IMAGE_API_ENV_FILE": str(config)}, clear=True):
                self.assertEqual(g.load_environment(None), [str(config.resolve())])
                self.assertEqual(g.base_url(), "https://openai.example")
                self.assertEqual(g.base_url(api_format="gemini"), "https://gemini.example")
                self.assertEqual(g.normalize_model(None), "CustomOpenAI")
                self.assertEqual(g.normalize_model(None, "gemini"), "CustomGemini")


if __name__ == "__main__":
    unittest.main()
