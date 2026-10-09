"""畸形响应与下载阶段回归；只使用假密钥、mock 和临时目录。"""

import base64
import io
import json
import os
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import requests

from test_generate_image import PNG_BYTES, generate_image as g
from image_gen.responses import decode_data_url, decode_image_base64


ENCODED_PNG = base64.b64encode(PNG_BYTES).decode("ascii")


class ResponseRegressionTests(unittest.TestCase):
    def setUp(self):
        for patcher in (
            patch.dict(os.environ, {
                "OPENAI_API_BASE_URL": "https://api.example",
                "OPENAI_API_KEY": "audit-fake-key",
                "GEMINI_API_BASE_URL": "https://gemini.example",
                "GEMINI_API_KEY": "audit-gemini-fake-key",
            }, clear=True),
            patch.object(g, "load_environment", return_value=[]),
            patch.object(requests.sessions.Session, "request", side_effect=AssertionError("Network forbidden")),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    def assert_invalid_response(self, callback, provider):
        with self.assertRaises(g.ImageAPIError) as caught:
            callback()
        self.assertEqual(caught.exception.code, "invalid_api_response")
        self.assertEqual(caught.exception.stage, "response")
        self.assertEqual(caught.exception.provider, provider)
        self.assertFalse(caught.exception.retryable)

    @staticmethod
    def openai(payload):
        return g.extract_openai_image(
            payload, "audit-fake-key", 5, "image/png", "https://api.example"
        )

    @staticmethod
    def gemini(payload):
        response = Mock(status_code=200, headers={})
        response.json.return_value = payload
        with patch.object(requests, "post", return_value=response):
            return g.call_gemini(
                "test", [], "16:9", "2K", "audit-gemini-fake-key", "https://gemini.example", 5,
                model="test-model",
            )

    def test_openai_nested_fields_reject_malformed_types(self):
        payloads = [
            None, [], {}, {"data": None}, {"data": {}}, {"data": [None]},
            {"data": [1]}, {"data": [{"b64_json": 123}]},
            {"data": [{"b64_json": ["not-a-string"]}]},
            {"data": [{"b64_json": None}]}, {"data": [{"b64_json": ""}]},
            {"data": [{"b64_json": "!invalid"}]}, {"data": [{"url": 123}]},
            {"data": [{"url": []}]}, {"data": [{"url": None}]},
        ]
        for payload in payloads:
            with self.subTest(payload=payload):
                self.assert_invalid_response(lambda: self.openai(payload), "openai")

    def test_openai_plain_base64_and_aliases_remain_supported(self):
        for field in ("b64_json", "base64", "image_b64"):
            with self.subTest(field=field):
                self.assertEqual(self.openai({"data": [{field: ENCODED_PNG}]}), (PNG_BYTES, "image/png"))

    def test_openai_nullable_base64_can_fall_back_to_url(self):
        with patch("image_gen.responses.download_image", return_value=(PNG_BYTES, "image/png")) as download:
            result = self.openai({"data": [{"b64_json": None, "url": "https://cdn.example/image.png"}]})
        self.assertEqual(result, (PNG_BYTES, "image/png"))
        download.assert_called_once()

    def test_base64_size_limit_is_checked_before_decode(self):
        with patch("image_gen.responses.MAX_REMOTE_IMAGE_BYTES", 3):
            self.assert_invalid_response(lambda: decode_image_base64("AAAA" * 3, "openai"), "openai")

    def test_data_url_rejects_non_image_or_invalid_encoding(self):
        values = [
            f"data:text/html;base64,{ENCODED_PNG}",
            f"data:image/gif;base64,{ENCODED_PNG}",
            f"data:;base64,{ENCODED_PNG}",
            "data:image/png;base64,", "data:image/png;base64,!invalid",
            f"data:image/png,{ENCODED_PNG}", "data:image/png;base64",
            f"data:image/png;not-base64,{ENCODED_PNG}",
        ]
        for value in values:
            with self.subTest(value=value):
                self.assert_invalid_response(lambda: self.openai({"data": [{"b64_json": value}]}), "openai")

    def test_data_url_accepts_supported_mime(self):
        for header in ("data:image/png;base64", "DATA:IMAGE/PNG;charset=utf-8;BASE64"):
            data_url = f"{header},{ENCODED_PNG}"
            with self.subTest(header=header):
                self.assertEqual(self.openai({"data": [{"b64_json": data_url}]}), (PNG_BYTES, "image/png"))
                self.assertEqual(decode_data_url(data_url, "openai"), (PNG_BYTES, "image/png"))
        self.assertIsNone(decode_data_url(ENCODED_PNG, "openai"))

    def test_gemini_nested_fields_reject_malformed_types(self):
        payloads = [
            [], {"candidates": None}, {"candidates": {}}, {"candidates": [None]},
            {"candidates": [{"content": None}]},
            {"candidates": [{"content": []}]},
            {"candidates": [{"content": {"parts": None}}]},
            {"candidates": [{"content": {"parts": {}}}]},
            {"candidates": [{"content": {"parts": [None]}}]},
        ]
        inline_values = [None, "invalid", [], {}, {"data": None}, {"data": 123},
                         {"data": ""}, {"data": "!invalid"},
                         {"data": ENCODED_PNG, "mimeType": None},
                         {"data": ENCODED_PNG, "mimeType": []},
                         {"data": ENCODED_PNG, "mimeType": "text/plain"}]
        payloads.extend({"candidates": [{"content": {"parts": [{"inlineData": inline}]}}]}
                        for inline in inline_values)
        for payload in payloads:
            with self.subTest(payload=payload):
                self.assert_invalid_response(lambda: self.gemini(payload), "gemini")

    def test_gemini_text_and_snake_case_inline_data_remain_supported(self):
        result = self.gemini({"candidates": [{"content": {"parts": [
            {"text": "Created image"},
            {"inline_data": {"data": ENCODED_PNG, "mime_type": "image/png"}},
        ]}}]})
        self.assertEqual(result.image, PNG_BYTES)
        self.assertEqual(result.mime_type, "image/png")

    def test_invalid_json_and_root_have_response_stage(self):
        response = Mock(status_code=200, text="bad json")
        response.json.side_effect = ValueError("bad json")
        self.assert_invalid_response(lambda: g.checked_json(response, "openai"), "openai")
        response.json.side_effect = None
        response.json.return_value = None
        self.assert_invalid_response(lambda: g.checked_json(response, "gemini"), "gemini")

    def test_cli_returns_json_for_malformed_provider_payload(self):
        with tempfile.TemporaryDirectory() as directory:
            for provider, payload in (
                ("openai", {"data": [{"b64_json": 123}]}),
                ("gemini", {"candidates": [None]}),
            ):
                with self.subTest(provider=provider):
                    response = Mock(status_code=200, headers={})
                    response.json.return_value = payload
                    stderr = io.StringIO()
                    argv = [g.__file__, "generate", "--api-format", provider,
                            "--prompt", "test", "--output-dir", directory]
                    with patch.object(requests, "post", return_value=response), \
                         patch.object(sys, "argv", argv), patch.object(sys, "stderr", stderr):
                        self.assertEqual(g.main(), 1)
                    error = json.loads(stderr.getvalue())
                    self.assertEqual(error["error_code"], "invalid_api_response")
                    self.assertEqual(error["stage"], "response")
                    self.assertEqual(error["provider"], provider)


class DownloadMetadataTests(unittest.TestCase):
    def setUp(self):
        patcher = patch.object(
            requests.sessions.Session, "request", side_effect=AssertionError("Network forbidden")
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    @staticmethod
    def download():
        return g.download_image(
            "https://cdn.example/image.png", "https://api.example", "audit-fake-key", 5
        )

    @staticmethod
    def response(status=200):
        response = Mock(status_code=status, headers={"Content-Type": "image/png"})
        response.url = "https://cdn.example/image.png"
        response.json.return_value = {"error": {"message": "download unavailable"}}
        response.iter_content.return_value = [PNG_BYTES]
        return response

    def test_download_http_errors_keep_status_and_retry_scope(self):
        for status, retryable in ((404, False), (429, True), (503, True)):
            with self.subTest(status=status):
                response = self.response(status)
                with patch.object(requests, "get", return_value=response), \
                     self.assertRaises(g.ImageAPIError) as caught:
                    self.download()
                error = caught.exception
                self.assertEqual(error.code, "image_download_http_error")
                self.assertEqual(error.stage, "download")
                self.assertEqual(error.provider, "openai")
                self.assertEqual(error.http_status, status)
                self.assertEqual(error.retryable, retryable)
                self.assertIn("不要重新调用生图 API", str(error))
                response.close.assert_called_once()

    def test_download_network_errors_keep_download_stage(self):
        for exception, code in ((requests.Timeout("test timeout"), "image_download_timeout"),
                                (requests.ConnectionError("test reset"), "image_download_network_error")):
            with self.subTest(code=code), patch.object(requests, "get", side_effect=exception), \
                 self.assertRaises(g.ImageAPIError) as caught:
                self.download()
            self.assertEqual(caught.exception.code, code)
            self.assertEqual(caught.exception.stage, "download")
            self.assertTrue(caught.exception.retryable)
            self.assertIsNone(caught.exception.http_status)

    def test_download_stream_network_error_preserves_received_status(self):
        response = self.response()
        response.iter_content.side_effect = requests.ConnectionError("test stream reset")
        with patch.object(requests, "get", return_value=response), \
             self.assertRaises(g.ImageAPIError) as caught:
            self.download()
        self.assertEqual(caught.exception.stage, "download")
        self.assertEqual(caught.exception.http_status, 200)
        response.close.assert_called_once()

    def test_download_invalid_url_is_structured(self):
        for url in ("https://cdn.example:invalid/image.png", "https://[bad/image.png", 1):
            with self.subTest(url=url), patch.object(requests, "get") as get, \
                 self.assertRaises(g.ImageAPIError) as caught:
                g.download_image(url, "https://api.example", "audit-fake-key", 5)
            self.assertEqual(caught.exception.code, "invalid_api_response")
            self.assertEqual(caught.exception.stage, "response")
            get.assert_not_called()

    def test_download_declared_non_image_mime_is_rejected(self):
        for mime in ("text/html", None, []):
            response = self.response()
            response.headers["Content-Type"] = mime
            with self.subTest(mime=mime), patch.object(requests, "get", return_value=response), \
                 self.assertRaises(g.ImageAPIError) as caught:
                self.download()
            self.assertEqual(caught.exception.code, "invalid_api_response")
            response.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
