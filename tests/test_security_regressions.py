"""安全与持久化回归：临时文件、假 key、禁止实际 HTTP。"""
import base64
from concurrent.futures import ThreadPoolExecutor
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

from PIL import Image
from test_generate_image import generate_image as g, PNG_BYTES as PNG, parse_args as args

FAKE_RESULT = (PNG, "image/png", "https://api.example/v1/images/generations")


class SecurityRegressions(unittest.TestCase):
    def setUp(self):
        self.process_env = {k: v for k, v in os.environ.items() if k.lower() in {"systemroot", "windir", "path", "temp", "tmp", "userprofile", "appdata", "localappdata"}}
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.env = patch.dict(os.environ, {"OPENAI_API_BASE_URL": "https://api.example", "OPENAI_API_KEY": "audit-fake-key"}, clear=True)
        self.env.start()
        self.addCleanup(self.env.stop)
        self.loader = patch.object(g, "load_environment", return_value=[])
        self.loader.start()
        self.addCleanup(self.loader.stop)
        network = patch.object(g.requests.sessions.Session, "request", side_effect=AssertionError("Network forbidden"))
        network.start()
        self.addCleanup(network.stop)

    def generate(self, *options):
        with patch.object(g, "call_openai", return_value=FAKE_RESULT):
            return g.run(args("generate", "--prompt", "audit", "--output-dir", str(self.root), *options))

    def test_cwd_env_cannot_redirect_credentials(self):
        self.loader.stop()
        os.environ.pop("OPENAI_API_BASE_URL", None)
        (self.root / ".env").write_text("OPENAI_API_BASE_URL=http://untrusted.example\n", encoding="utf-8")
        trusted = self.root / "trusted"
        trusted.mkdir()
        (trusted / ".env").write_text("OPENAI_API_BASE_URL=https://api.example\n", encoding="utf-8")
        response = Mock(status_code=200)
        response.headers = {}
        response.json.return_value = {"data": [{"b64_json": base64.b64encode(PNG).decode()}]}
        with patch.object(Path, "cwd", return_value=self.root), patch.object(g, "__file__", str(trusted / "scripts/generate_image.py")), patch.object(g.requests, "post", return_value=response) as post:
            g.run(args("generate", "--prompt", "audit", "--output-dir", str(self.root)))
        self.assertEqual(post.call_args.args[0], "https://api.example/v1/images/generations")
        self.assertFalse(post.call_args.kwargs["allow_redirects"])
        self.assertEqual(post.call_args.kwargs["headers"]["Authorization"], "Bearer audit-fake-key")

    def test_explicit_configuration_is_supported(self):
        self.loader.stop()
        os.environ.pop("OPENAI_API_BASE_URL", None)
        configuration = self.root / ".env"
        configuration.write_text("OPENAI_API_BASE_URL=https://chosen.example\n", encoding="utf-8")
        result = g.run(args("generate", "--prompt", "audit", "--env-file", str(configuration), "--dry-run"))
        self.assertTrue(result["endpoint"].startswith("https://chosen.example/"))

    def test_unsafe_api_endpoints_rejected_before_request(self):
        for url in ("http://api.example", "https://user:pass@api.example", "https://api.example?key=x", "https://api.example#fragment", "https://api.example:bad", "https:///missing"):
            with self.subTest(url=url), patch.dict(os.environ, {"OPENAI_API_BASE_URL": url}), patch.object(g, "call_openai") as api:
                with self.assertRaises(ValueError):
                    g.run(args("generate", "--prompt", "audit", "--output-dir", str(self.root)))
                api.assert_not_called()

    def test_parallel_manifest_preserves_both_assets(self):
        manifest = self.root / "manifest.json"
        def worker(asset):
            return g.run(args("generate", "--prompt", "audit", "--manifest", str(manifest), "--asset-id", asset))
        with patch.object(g, "call_openai", return_value=FAKE_RESULT), ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(worker, ["card-01", "card-02"]))
        self.assertEqual(len(results), 2)
        self.assertEqual(set(json.loads(manifest.read_text())["assets"]), {"card-01", "card-02"})

    def test_duplicate_asset_concurrency_calls_api_once(self):
        manifest = self.root / "manifest.json"
        def worker(_):
            try:
                return g.run(args("generate", "--prompt", "audit", "--manifest", str(manifest), "--asset-id", "same"))
            except ValueError as exc:
                return str(exc)
        with patch.object(g, "call_openai", return_value=FAKE_RESULT) as api, ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(worker, range(2)))
        self.assertEqual(api.call_count, 1)
        self.assertEqual(sum(isinstance(v, dict) for v in results), 1)

    def test_same_session_edits_preserve_history(self):
        first = self.generate()
        def worker(prompt):
            return g.run(args("edit", "--prompt", prompt, "--session", first["session"]))
        with patch.object(g, "call_openai", return_value=FAKE_RESULT), ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(worker, ["change A", "change B"]))
        session = json.loads(Path(first["session"]).read_text(encoding="utf-8"))
        self.assertEqual(len(session["turns"]), 3)
        self.assertEqual(session["turns"][2]["input_image"], session["turns"][1]["output_image"])

    def test_cross_process_manifest_lock(self):
        # 子进程使用相同真实 FileLock；不依赖当前测试进程的 mock。
        code = '''import base64, importlib.util, os, sys, time
from unittest.mock import patch
from pathlib import Path
spec = importlib.util.spec_from_file_location("worker", sys.argv[1])
g = importlib.util.module_from_spec(spec)
spec.loader.exec_module(g)
data = base64.b64decode(sys.argv[4])
def fake(*unused, **kwargs):
    time.sleep(0.2)
    return data, "image/png", "https://api.example"
with patch.dict(os.environ, {"OPENAI_API_BASE_URL":"https://api.example", "OPENAI_API_KEY":"fake"}, clear=True), patch.object(g,"load_environment",return_value=[]), patch.object(g,"call_openai",side_effect=fake), patch.object(g.requests.sessions.Session,"request",side_effect=AssertionError("network forbidden")):
    g.run(g.build_parser().parse_args(["generate","--prompt","audit","--manifest",sys.argv[2],"--asset-id",sys.argv[3]]))
'''
        manifest = self.root / "processes.json"
        processes = [subprocess.Popen(
            [sys.executable, "-X", "utf8", "-c", code, g.__file__, str(manifest), asset, base64.b64encode(PNG).decode()],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            env=self.process_env,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        ) for asset in ("one", "two")]
        outcomes = []
        try:
            for process in processes:
                _, stderr = process.communicate(timeout=15)
                outcomes.append((process.returncode, stderr.decode(errors="replace")))
        finally:
            for process in processes:
                if process.poll() is None:
                    process.kill()
                process.communicate()
        for returncode, stderr in outcomes:
            self.assertEqual(returncode, 0, stderr)
        self.assertEqual(set(json.loads(manifest.read_text())["assets"]), {"one", "two"})

    def test_output_session_collision_rejected_before_api(self):
        same = self.root / "result.png"
        with patch.object(g, "call_openai") as api:
            with self.assertRaises(ValueError):
                g.run(args("generate", "--prompt", "audit", "--output", str(same), "--session", str(same)))
            api.assert_not_called()
        self.assertFalse(same.exists())

    def test_session_manifest_collision_rejected_without_lock_wait(self):
        same = self.root / "both.json"
        with patch.object(g, "call_openai") as api:
            with self.assertRaisesRegex(ValueError, "不能相同"):
                g.run(args("generate", "--prompt", "audit", "--manifest", str(same), "--asset-id", "x", "--session", str(same)))
            api.assert_not_called()

    def test_atomic_image_publication_never_overwrites_existing_file(self):
        target = self.root / "keep.png"
        target.write_bytes(b"user work")
        with self.assertRaises(FileExistsError):
            g.write_image_exclusive(target, PNG)
        self.assertEqual(target.read_bytes(), b"user work")

    def test_truncated_png_is_rejected_without_manifest(self):
        manifest = self.root / "manifest.json"
        with patch.object(g, "call_openai", return_value=(PNG[:8], "image/png", "https://api.example")):
            with self.assertRaises(g.ImageAPIError):
                g.run(args("generate", "--prompt", "audit", "--manifest", str(manifest), "--asset-id", "broken"))
        self.assertFalse(manifest.exists())

    def test_truncated_reference_is_rejected_before_api(self):
        source = self.root / "broken.png"
        source.write_bytes(PNG[:8])
        with patch.object(g, "call_openai") as api:
            with self.assertRaises(ValueError):
                g.run(args("reference", "--prompt", "audit", "--image", str(source)))
            api.assert_not_called()

    def test_valid_formats_are_decoded_and_pixel_limit_is_checked(self):
        for format_name, mime in (("PNG", "image/png"), ("JPEG", "image/jpeg"), ("WEBP", "image/webp")):
            with self.subTest(format_name=format_name):
                buffer = io.BytesIO()
                Image.new("RGB", (64, 36)).save(buffer, format=format_name)
                self.assertEqual(g.validated_image_extension(buffer.getvalue(), mime), format_name.lower())
        with patch.object(g, "MAX_IMAGE_PIXELS", 10):
            with self.assertRaises(g.ImageAPIError):
                g.validated_image_extension(PNG, "image/png")

    def test_generated_status_is_not_visual_approval(self):
        manifest = self.root / "manifest.json"
        result = self.generate("--manifest", str(manifest), "--asset-id", "card")
        self.assertEqual(result["actual_dimensions"], {"width": 64, "height": 36})
        self.assertEqual(json.loads(manifest.read_text())["assets"]["card"]["status"], "generated")

    def test_manifest_failure_blocks_retry_and_recovers_without_api(self):
        manifest = self.root / "manifest.json"
        first = self.generate("--manifest", str(manifest), "--asset-id", "card", "--output", "v1.png")
        original = manifest.read_bytes()
        atomic_write = g.write_json_atomic
        def fail(path, payload):
            if path == manifest:
                raise PermissionError("simulated manifest write failure")
            return atomic_write(path, payload)
        with patch.object(g, "call_openai", return_value=FAKE_RESULT), patch.object(g, "write_json_atomic", side_effect=fail):
            with self.assertRaisesRegex(g.ImageAPIError, "recover --journal"):
                g.run(args("edit", "--prompt", "change", "--manifest", str(manifest), "--asset-id", "card", "--output", "v2.png"))
        self.assertEqual(manifest.read_bytes(), original)
        with patch.object(g, "call_openai") as api:
            for options in (("--manifest", str(manifest), "--asset-id", "card"), ("--session", first["session"])):
                with self.assertRaisesRegex(g.ImageAPIError, "恢复"):
                    g.run(args("edit", "--prompt", "retry", *options))
            recovered = g.run(args("recover", "--journal", str(g.pending_path(manifest))))
            api.assert_not_called()
        self.assertEqual(Path(recovered["image"]).name, "v2.png")
        self.assertEqual(len(json.loads(Path(first["session"]).read_text(encoding="utf-8"))["turns"]), 2)
        self.assertEqual(json.loads(manifest.read_text())["assets"]["card"]["image"], "v2.png")
        self.assertFalse(g.pending_path(manifest).exists())
        plan = g.run(args("edit", "--prompt", "next", "--manifest", str(manifest), "--asset-id", "card", "--dry-run"))
        self.assertEqual(Path(plan["source_image"]).name, "v2.png")

    def test_image_write_failure_can_recover_from_journal(self):
        session = self.root / "session.json"
        with patch.object(g, "write_image_exclusive", side_effect=OSError("disk unavailable")):
            with self.assertRaisesRegex(g.ImageAPIError, "recover --journal"):
                self.generate("--session", str(session))
        with patch.object(g, "call_openai") as api:
            result = g.run(args("recover", "--journal", str(g.pending_path(session))))
            api.assert_not_called()
        self.assertEqual(Path(result["image"]).read_bytes(), PNG)

    def test_session_write_failure_and_repeated_recovery_are_safe(self):
        session = self.root / "session.json"
        atomic_write = g.write_json_atomic
        def fail(path, payload):
            if path == session:
                raise PermissionError("session unavailable")
            return atomic_write(path, payload)
        with patch.object(g, "write_json_atomic", side_effect=fail):
            with self.assertRaisesRegex(g.ImageAPIError, "recover --journal"):
                self.generate("--session", str(session))
            with self.assertRaises(g.ImageAPIError) as error:
                g.run(args("recover", "--journal", str(g.pending_path(session))))
            self.assertEqual(error.exception.code, "storage_pending")
            self.assertEqual(error.exception.stage, "storage")
        self.assertTrue(g.pending_path(session).exists())
        result = g.run(args("recover", "--journal", str(g.pending_path(session))))
        self.assertEqual(len(json.loads(session.read_text(encoding="utf-8"))["turns"]), 1)
        self.assertEqual(Path(result["image"]).read_bytes(), PNG)

    def test_recovery_does_not_overwrite_different_image(self):
        session = self.root / "session.json"
        with patch.object(g, "write_image_exclusive", side_effect=OSError("disk unavailable")):
            with self.assertRaises(g.ImageAPIError):
                self.generate("--session", str(session))
        journal = g.pending_path(session)
        target = Path(json.loads(journal.read_text(encoding="utf-8"))["result"]["image"])
        target.write_bytes(b"user replacement")
        with self.assertRaisesRegex(ValueError, "拒绝覆盖"):
            g.run(args("recover", "--journal", str(journal)))
        self.assertEqual(target.read_bytes(), b"user replacement")
        self.assertTrue(journal.exists())

    def test_corrupt_recovery_log_returns_value_error(self):
        journal = self.root / "session.json.pending.json"
        journal.write_text(json.dumps({"version": 1, "result": {"image": "image.png", "session": "session.json", "mime_type": "image/png"}}), encoding="utf-8")
        with self.assertRaises(ValueError):
            g.run(args("recover", "--journal", str(journal)))

    def test_dry_run_creates_no_files(self):
        before = list(self.root.rglob("*"))
        g.run(args("generate", "--prompt", "audit", "--manifest", str(self.root / "manifest.json"), "--asset-id", "card", "--dry-run"))
        self.assertEqual(list(self.root.rglob("*")), before)

    def test_unsupported_output_filesystem_rejected_before_api(self):
        with patch.object(os, "link", side_effect=OSError("unsupported filesystem")), patch.object(g, "call_openai") as api:
            with self.assertRaisesRegex(ValueError, "硬链接"):
                g.run(args("generate", "--prompt", "audit", "--output-dir", str(self.root)))
            api.assert_not_called()

    def test_manifest_owned_session_requires_manifest_edit(self):
        first = self.generate("--manifest", str(self.root / "manifest.json"), "--asset-id", "card")
        with self.assertRaisesRegex(ValueError, "属于 manifest"):
            g.run(args("edit", "--prompt", "audit", "--session", first["session"], "--dry-run"))

    def test_moved_bundle_edits_using_relative_paths(self):
        old = self.root / "old"
        new = self.root / "new"
        with patch.object(g, "call_openai", return_value=FAKE_RESULT):
            g.run(args("generate", "--prompt", "audit", "--manifest", str(old / "manifest.json"), "--asset-id", "card"))
        old.rename(new)
        result = g.run(args("edit", "--prompt", "audit", "--manifest", str(new / "manifest.json"), "--asset-id", "card", "--dry-run"))
        self.assertEqual(Path(result["source_image"]).parent, new)

    def test_legacy_absolute_session_paths_still_work(self):
        first = self.generate()
        session_path = Path(first["session"])
        session = json.loads(session_path.read_text(encoding="utf-8"))
        session["turns"][-1]["output_image"] = first["image"]
        session_path.write_text(json.dumps(session), encoding="utf-8")
        result = g.run(args("edit", "--prompt", "audit", "--session", str(session_path), "--dry-run"))
        self.assertEqual(result["source_image"], first["image"])

    def test_invalid_session_fields_produce_value_errors(self):
        session = self.root / "session.json"
        variants = [
            {"parameters": None}, {"parameters": {"quality": []}},
            {"parameters": {"image_size": "8K"}}, {"parameters": {"size": 5}},
            {"parameters": {"compression": "100"}}, {"model": {}},
            {"context": None}, {"context": [1]}, {"turns": None},
            {"turns": [None]}, {"turns": [{"prompt": 1}]},
            {"turns": [{"output_image": []}]}, {"manifest": 1},
        ]
        for fields in variants:
            with self.subTest(fields=fields):
                session.write_text(json.dumps({"version": 1, **fields}), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "会话字段无效"):
                    g.run(args("edit", "--prompt", "audit", "--session", str(session), "--dry-run"))

    def test_invalid_session_cli_returns_json_error(self):
        session = self.root / "session.json"
        session.write_text('{"version":1,"parameters":null}', encoding="utf-8")
        stderr = io.StringIO()
        with patch.object(sys, "argv", [g.__file__, "edit", "--session", str(session), "--prompt", "audit", "--dry-run"]), patch.object(sys, "stderr", stderr):
            self.assertEqual(g.main(), 1)
        payload = json.loads(stderr.getvalue())
        self.assertIn("parameters", payload["error"])
        self.assertEqual(payload["error_code"], "validation_error")
        self.assertEqual(payload["stage"], "validation")
        self.assertEqual(payload["provider"], "openai")
        self.assertFalse(payload["retryable"])
        self.assertIsNone(payload["http_status"])


if __name__ == "__main__":
    unittest.main()
