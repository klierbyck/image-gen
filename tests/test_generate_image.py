import importlib.util
import io
from PIL import Image
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "generate_image.py"
SPEC = importlib.util.spec_from_file_location("generate_image", SCRIPT)
assert SPEC and SPEC.loader
generate_image = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(generate_image)
BUFFER = io.BytesIO()
Image.new("RGB", (64, 36), "white").save(BUFFER, format="PNG")
PNG_BYTES = BUFFER.getvalue()


def parse_args(*values):
    return generate_image.build_parser().parse_args(list(values))


def run_with_fake_openai(args):
    with patch.dict(os.environ, {"OPENAI_API_BASE_URL": "https://api.example", "OPENAI_API_KEY": "test-key"}, clear=True):
        with patch.object(
            generate_image,
            "call_openai",
            return_value=(PNG_BYTES, "image/png", "https://example.test/v1/images"),
        ) as mocked:
            return generate_image.run(args), mocked


class OutputFormatTests(unittest.TestCase):
    def test_detects_supported_image_signatures(self):
        self.assertEqual(generate_image.detected_image_extension(b"\x89PNG\r\n\x1a\nrest"), "png")
        self.assertEqual(generate_image.detected_image_extension(b"\xff\xd8\xffrest"), "jpeg")
        self.assertEqual(generate_image.detected_image_extension(b"RIFF\x00\x00\x00\x00WEBPrest"), "webp")

    def test_rejects_non_image_data(self):
        with self.assertRaises(generate_image.ImageAPIError):
            generate_image.validated_image_extension(b"not an image", "text/html")

    def test_rejects_mime_signature_mismatch(self):
        with self.assertRaises(generate_image.ImageAPIError):
            generate_image.validated_image_extension(
                b"\x89PNG\r\n\x1a\nrest", "image/jpeg"
            )

    def test_requested_path_uses_actual_extension(self):
        with tempfile.TemporaryDirectory() as directory:
            result = generate_image.requested_output_path(
                Path(directory), "result.jpeg", "png"
            )
            self.assertEqual(result.suffix, ".png")

    def test_jpg_alias_is_preserved_for_jpeg(self):
        with tempfile.TemporaryDirectory() as directory:
            result = generate_image.requested_output_path(
                Path(directory), "result.jpg", "jpeg"
            )
            self.assertEqual(result.suffix, ".jpg")

    def test_rejects_non_image_source_file(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "not-an-image.png"
            source.write_text("not an image", encoding="utf-8")
            args = parse_args(
                "reference",
                "--image",
                str(source),
                "--prompt",
                "test",
                "--dry-run",
            )
            with patch.dict(os.environ, {"OPENAI_API_BASE_URL": "https://api.example"}, clear=True):
                with self.assertRaisesRegex(ValueError, "不是受支持"):
                    generate_image.run(args)


class GeminiValidationTests(unittest.TestCase):
    def test_gemini_rejects_non_png_output_before_api_call(self):
        parser = generate_image.build_parser()
        args = parser.parse_args([
            "generate",
            "--api-format", "gemini",
            "--model",
            "nana-banana-2",
            "--prompt",
            "test",
            "--output-format",
            "jpeg",
            "--dry-run",
        ])
        with patch.dict(os.environ, {"OPENAI_API_BASE_URL": "https://api.example"}, clear=True):
            with self.assertRaisesRegex(ValueError, "只输出 PNG"):
                generate_image.run(args)

    def test_gemini_rejects_openai_only_quality_and_compression(self):
        for option, value in (("--quality", "high"), ("--compression", "10")):
            with self.subTest(option=option):
                args = parse_args(
                    "generate",
                    "--api-format", "gemini",
                    "--model",
                    "nana-banana-2",
                    "--prompt",
                    "test",
                    option,
                    value,
                    "--dry-run",
                )
                with patch.dict(os.environ, {"OPENAI_API_BASE_URL": "https://api.example"}, clear=True):
                    with self.assertRaisesRegex(ValueError, "仅适用于 OpenAI Images"):
                        generate_image.run(args)


class ModelSelectionTests(unittest.TestCase):
    def test_environment_sets_default_model(self):
        with patch.dict(os.environ, {"OPENAI_API_BASE_URL": "https://api.example", "OPENAI_MODEL": "nana-banana-2"}, clear=True):
            self.assertEqual(generate_image.normalize_model(None), "nana-banana-2")

    def test_missing_environment_uses_openai_default(self):
        with patch.dict(os.environ, {"OPENAI_API_BASE_URL": "https://api.example"}, clear=True):
            self.assertEqual(generate_image.normalize_model(None), "gpt-image-2")

    def test_explicit_model_overrides_environment_default(self):
        with patch.dict(os.environ, {"OPENAI_API_BASE_URL": "https://api.example", "OPENAI_MODEL": "nana-banana-2"}, clear=True):
            self.assertEqual(
                generate_image.normalize_model("gpt-image-2"), "gpt-image-2"
            )

    def test_session_model_overrides_environment_default(self):
        parser = generate_image.build_parser()
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.png"
            source.write_bytes(PNG_BYTES)
            session = Path(directory) / "session.json"
            session.write_text(
                '{"version": 1, "model": "gpt-image-2", "parameters": {}, "turns": []}',
                encoding="utf-8",
            )
            args = parser.parse_args([
                "edit",
                "--session",
                str(session),
                "--image",
                str(source),
                "--prompt",
                "test",
                "--dry-run",
            ])
            with patch.dict(os.environ, {"OPENAI_API_BASE_URL": "https://api.example", "OPENAI_MODEL": "nana-banana-2"}, clear=True):
                result = generate_image.run(args)
            self.assertEqual(result["model"], "gpt-image-2")


class IndependentModeTests(unittest.TestCase):
    def run_dry(self, arguments):
        parser = generate_image.build_parser()
        args = parser.parse_args(arguments + ["--dry-run"])
        with patch.dict(os.environ, {"OPENAI_API_BASE_URL": "https://api.example"}, clear=True):
            return generate_image.run(args)

    def test_generate_remains_independent(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self.run_dry([
                "generate",
                "--model",
                "gpt-image-2",
                "--prompt",
                "standalone image",
                "--output-dir",
                directory,
            ])
            self.assertEqual(result["command"], "generate")
            self.assertEqual(result["source_images"], [])

    def test_reference_remains_independent(self):
        with tempfile.TemporaryDirectory() as directory:
            reference = Path(directory) / "reference.png"
            reference.write_bytes(PNG_BYTES)
            result = self.run_dry([
                "reference",
                "--model",
                "gpt-image-2",
                "--image",
                str(reference),
                "--prompt",
                "new composition",
                "--output-dir",
                directory,
            ])
            self.assertEqual(result["command"], "reference")
            self.assertEqual(result["source_images"], [str(reference.resolve())])

    def test_edit_remains_independent_without_platform_skill(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.png"
            source.write_bytes(PNG_BYTES)
            result = self.run_dry([
                "edit",
                "--model",
                "gpt-image-2",
                "--image",
                str(source),
                "--prompt",
                "change the background",
                "--output-dir",
                directory,
            ])
            self.assertEqual(result["command"], "edit")
            self.assertEqual(result["source_image"], str(source.resolve()))

    def test_generate_and_reference_without_manifest_remain_supported(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "reference.png"
            source.write_bytes(PNG_BYTES)
            for values in (
                ("generate", "--prompt", "standalone", "--dry-run"),
                (
                    "reference",
                    "--image",
                    str(source),
                    "--prompt",
                    "standalone reference",
                    "--dry-run",
                ),
                ):
                    with self.subTest(command=values[0]):
                        with patch.dict(os.environ, {"OPENAI_API_BASE_URL": "https://api.example"}, clear=True):
                            result = generate_image.run(parse_args(*values))
                        self.assertIsNone(result["manifest"])
                        self.assertIsNone(result["asset_id"])


class SizeValidationTests(unittest.TestCase):
    def test_exact_size_records_matching_aspect_ratio(self):
        args = parse_args(
            "generate",
            "--prompt",
            "square",
            "--size",
            "1024x1024",
            "--dry-run",
        )
        with patch.dict(os.environ, {"OPENAI_API_BASE_URL": "https://api.example"}, clear=True):
            result = generate_image.run(args)
        self.assertEqual(result["parameters"]["aspect_ratio"], "1:1")
        self.assertEqual(result["parameters"]["size"], "1024x1024")
        self.assertIsNone(result["parameters"]["image_size"])

    def test_exact_size_rejects_ratio_or_size_tier(self):
        for option, value in (("--aspect-ratio", "1:1"), ("--image-size", "1K")):
            with self.subTest(option=option):
                args = parse_args(
                    "generate",
                    "--prompt",
                    "conflict",
                    "--size",
                    "1024x1024",
                    option,
                    value,
                    "--dry-run",
                )
                with patch.dict(os.environ, {"OPENAI_API_BASE_URL": "https://api.example"}, clear=True):
                    with self.assertRaisesRegex(ValueError, "不能与"):
                        generate_image.run(args)


class ManifestTests(unittest.TestCase):
    def test_manifest_and_asset_id_are_required_together(self):
        for option, value in (
            ("--manifest", "generation-manifest.json"),
            ("--asset-id", "card-01"),
        ):
            with self.subTest(option=option):
                args = parse_args(
                    "generate",
                    "--prompt",
                    "incomplete manifest arguments",
                    option,
                    value,
                    "--dry-run",
                )
                with patch.dict(os.environ, {"OPENAI_API_BASE_URL": "https://api.example"}, clear=True):
                    with self.assertRaisesRegex(ValueError, "必须同时使用"):
                        generate_image.run(args)

    def test_manifest_create_and_edit_reuses_asset_session(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = root / "generation-manifest.json"
            prompt_file = root / "prompt.md"
            prompt_file.write_text("first prompt", encoding="utf-8")
            create_args = parse_args(
                "generate",
                "--prompt-file",
                str(prompt_file),
                "--manifest",
                str(manifest_path),
                "--asset-id",
                "card-01",
                "--asset-role",
                "cover",
                "--output",
                "card-01.png",
            )
            created, _ = run_with_fake_openai(create_args)

            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            asset = manifest["assets"]["card-01"]
            self.assertEqual(asset["image"], "card-01.png")
            self.assertEqual(asset["prompt_file"], "prompt.md")
            self.assertEqual(asset["status"], "generated")
            first_session = created["session"]

            edit_args = parse_args(
                "edit",
                "--prompt",
                "second prompt",
                "--manifest",
                str(manifest_path),
                "--asset-id",
                "card-01",
                "--output",
                "card-01-v2.png",
            )
            edited, mocked = run_with_fake_openai(edit_args)
            self.assertEqual(edited["session"], first_session)
            self.assertEqual(mocked.call_args.args[1], [Path(created["image"])])

            updated = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(updated["assets"]["card-01"]["image"], "card-01-v2.png")
            self.assertEqual(updated["assets"]["card-01"]["action"], "edit")
            session = json.loads(Path(first_session).read_text(encoding="utf-8"))
            self.assertEqual(len(session["turns"]), 2)

    def test_duplicate_asset_id_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest_path = Path(directory) / "generation-manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "version": 1,
                        "assets": {"card-01": {"status": "complete"}},
                    }
                ),
                encoding="utf-8",
            )
            args = parse_args(
                "generate",
                "--prompt",
                "duplicate",
                "--manifest",
                str(manifest_path),
                "--asset-id",
                "card-01",
                "--dry-run",
            )
            with patch.dict(os.environ, {"OPENAI_API_BASE_URL": "https://api.example"}, clear=True):
                with self.assertRaisesRegex(ValueError, "asset_id 已存在"):
                    generate_image.run(args)

    def test_invalid_asset_record_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest_path = Path(directory) / "generation-manifest.json"
            manifest_path.write_text(
                json.dumps({"version": 1, "assets": {"card-01": "broken"}}),
                encoding="utf-8",
            )
            args = parse_args(
                "edit",
                "--prompt",
                "edit",
                "--manifest",
                str(manifest_path),
                "--asset-id",
                "card-01",
                "--dry-run",
            )
            with patch.dict(os.environ, {"OPENAI_API_BASE_URL": "https://api.example"}, clear=True):
                with self.assertRaisesRegex(ValueError, "资产记录无效"):
                    generate_image.run(args)

    def test_reference_asset_id_supplies_anchor_image(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            anchor = root / "card-01.png"
            anchor.write_bytes(PNG_BYTES)
            manifest_path = root / "generation-manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "version": 1,
                        "assets": {
                            "card-01": {
                                "image": "card-01.png",
                                "status": "complete",
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            args = parse_args(
                "reference",
                "--prompt",
                "continue the visual language",
                "--manifest",
                str(manifest_path),
                "--asset-id",
                "card-02",
                "--reference-asset-id",
                "card-01",
                "--output",
                "card-02.png",
            )
            result, mocked = run_with_fake_openai(args)
            self.assertEqual(mocked.call_args.args[1], [anchor.resolve()])
            self.assertEqual(result["asset_id"], "card-02")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(
                manifest["assets"]["card-02"]["reference_asset_id"], "card-01"
            )


class DownloadSecurityTests(unittest.TestCase):
    @staticmethod
    def response(data=PNG_BYTES, **headers):
        response = Mock()
        response.status_code = 200
        response.headers = headers
        response.url = "https://cdn.example/image.png"
        response.iter_content.return_value = [data]
        return response

    def test_cross_origin_download_omits_authorization(self):
        response = self.response(**{"Content-Type": "image/png"})
        with patch.object(generate_image.requests, "get", return_value=response) as get:
            data, _ = generate_image.download_image(
                "https://cdn.example/image.png",
                "https://api.example",
                "secret",
                5,
            )
        self.assertEqual(data, PNG_BYTES)
        self.assertEqual(get.call_args.kwargs["headers"], {})

    def test_same_origin_download_sends_authorization(self):
        response = self.response(**{"Content-Type": "image/png"})
        response.url = "https://api.example/image.png"
        with patch.object(generate_image.requests, "get", return_value=response) as get:
            generate_image.download_image(
                "https://api.example:443/image.png",
                "https://api.example",
                "secret",
                5,
            )
        self.assertEqual(
            get.call_args.kwargs["headers"], {"Authorization": "Bearer secret"}
        )

    def test_external_http_download_is_rejected(self):
        with patch.object(generate_image.requests, "get") as get:
            with self.assertRaisesRegex(generate_image.ImageAPIError, "非 HTTPS"):
                generate_image.download_image(
                    "http://cdn.example/image.png",
                    "https://api.example",
                    "secret",
                    5,
                )
        get.assert_not_called()

    def test_remote_image_size_limit_is_enforced(self):
        response = self.response(
            **{
                "Content-Type": "image/png",
                "Content-Length": str(generate_image.MAX_REMOTE_IMAGE_BYTES + 1),
            }
        )
        with patch.object(generate_image.requests, "get", return_value=response):
            with self.assertRaisesRegex(generate_image.ImageAPIError, "32 MB"):
                generate_image.download_image(
                    "https://cdn.example/image.png",
                    "https://api.example",
                    "secret",
                    5,
                )

    def test_streamed_body_size_limit_is_enforced(self):
        response = self.response(**{"Content-Type": "image/png"})
        response.iter_content.return_value = [
            b"x" * generate_image.MAX_REMOTE_IMAGE_BYTES,
            b"x",
        ]
        with patch.object(generate_image.requests, "get", return_value=response):
            with self.assertRaisesRegex(generate_image.ImageAPIError, "32 MB"):
                generate_image.download_image(
                    "https://cdn.example/image.png",
                    "https://api.example",
                    "secret",
                    5,
                )


if __name__ == "__main__":
    unittest.main()
