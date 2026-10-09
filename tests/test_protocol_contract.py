"""Provider protocol tests against an in-process HTTP server; no external network."""

import base64
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from test_generate_image import PNG_BYTES, generate_image as g


class ProtocolHandler(BaseHTTPRequestHandler):
    requests = []

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)
        self.__class__.requests.append({
            "path": self.path,
            "headers": dict(self.headers),
            "body": body,
        })
        encoded = base64.b64encode(PNG_BYTES).decode("ascii")
        if ":generateContent" in self.path:
            payload = {
                "candidates": [{"content": {"parts": [{"inlineData": {
                    "mimeType": "image/png",
                    "data": encoded,
                }}]}}],
                "usageMetadata": {"totalTokenCount": 9},
            }
        else:
            payload = {
                "data": [{"b64_json": encoded}],
                "usage": {"total_tokens": 7},
            }
        data = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("x-request-id", "local-request")
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, _format, *_args):
        pass


class ProtocolContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ProtocolHandler.requests = []
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), ProtocolHandler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base_url = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=5)

    def setUp(self):
        ProtocolHandler.requests.clear()
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.source = Path(directory.name) / "source.png"
        self.source.write_bytes(PNG_BYTES)

    def test_openai_json_generation_contract(self):
        with patch.object(g, "validate_api_url"):
            result = g.call_openai(
                "local generation", [], "2048x1152", "high", "png", 100,
                "fake-key", self.base_url, 5, model="OpenAIImage-v3",
            )
        request = ProtocolHandler.requests[-1]
        payload = json.loads(request["body"])
        self.assertEqual(request["path"], "/v1/images/generations")
        self.assertEqual(payload["model"], "OpenAIImage-v3")
        self.assertEqual(payload["prompt"], "local generation")
        self.assertEqual(request["headers"]["Authorization"], "Bearer fake-key")
        self.assertEqual(result.request_id, "local-request")
        self.assertEqual(result.usage, {"total_tokens": 7})

    def test_openai_multipart_edit_contract(self):
        with patch.object(g, "validate_api_url"):
            result = g.call_openai(
                "local edit", [self.source], "2048x1152", "auto", "png", 100,
                "fake-key", self.base_url, 5, model="OpenAIImage-v3",
            )
        request = ProtocolHandler.requests[-1]
        self.assertEqual(request["path"], "/v1/images/edits")
        self.assertIn("multipart/form-data", request["headers"]["Content-Type"])
        self.assertIn(b'local edit', request["body"])
        self.assertIn(b'filename="source.png"', request["body"])
        self.assertEqual(result.image, PNG_BYTES)

    def test_gemini_inline_data_contract(self):
        with patch.object(g, "validate_api_url"):
            result = g.call_gemini(
                "local reference", [self.source], "3:4", "2K", "fake-key",
                self.base_url, 5, model="GeminiImage-v3",
            )
        request = ProtocolHandler.requests[-1]
        payload = json.loads(request["body"])
        self.assertEqual(
            request["path"], "/v1beta/models/GeminiImage-v3:generateContent"
        )
        parts = payload["contents"][0]["parts"]
        self.assertEqual(base64.b64decode(parts[1]["inlineData"]["data"]), PNG_BYTES)
        self.assertEqual(payload["generationConfig"]["imageConfig"], {
            "aspectRatio": "3:4", "imageSize": "2K",
        })
        self.assertEqual(result.usage, {"totalTokenCount": 9})


if __name__ == "__main__":
    unittest.main()
