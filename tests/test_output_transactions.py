"""显式输出并发与改存恢复：真实文件锁、临时目录、假 key，禁止外网。"""

from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import io
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

from test_generate_image import generate_image as g, PNG_BYTES, parse_args
from image_gen.output_locks import acquire_output_locks, output_lock_paths


class OutputLockTests(unittest.TestCase):
    def test_paths_are_resolved_sorted_and_deduplicated(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first, second = root / "a.png", root / "b.png"
            paths = output_lock_paths([second, first, root / "nested" / ".." / "a.png"])
            self.assertEqual(len(paths), 2)
            self.assertEqual(paths, output_lock_paths([first, second]))
            self.assertTrue(all(path.is_absolute() for path in paths))
            self.assertEqual(paths, sorted(paths, key=lambda path: os.path.normcase(str(path))))

    @unittest.skipUnless(os.name == "nt", "Windows paths are case insensitive")
    def test_windows_case_aliases_use_one_lock(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertEqual(len(output_lock_paths([root / "same.png", root / "SAME.PNG"])), 1)

    def test_dry_run_creates_no_directory_or_sidecar(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "not-created"
            with ExitStack() as stack:
                acquire_output_locks(stack, [root / "image.png"], True)
            self.assertFalse(root.exists())

    def test_locks_exclude_another_process_until_stack_exits(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = [root / "a.png", root / "b.png"]
            code = (
                "from contextlib import ExitStack; from pathlib import Path; import sys; "
                "sys.path.insert(0, sys.argv[1]); "
                "from image_gen.output_locks import acquire_output_locks; "
                "print('ready', flush=True); "
                "stack = ExitStack(); "
                "acquire_output_locks(stack, [Path(p) for p in sys.argv[2:]], False); "
                "print('acquired', flush=True); stack.close()"
            )
            output = queue.Queue()
            process = None
            try:
                with ExitStack() as stack:
                    acquire_output_locks(stack, paths, False)
                    process = subprocess.Popen(
                        [sys.executable, "-X", "utf8", "-c", code,
                         str(Path(g.__file__).parent), *(str(path) for path in reversed(paths))],
                        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                    )
                    def read_output():
                        for line in process.stdout:
                            output.put(line.strip())
                    reader = threading.Thread(target=read_output, daemon=True)
                    reader.start()
                    self.assertEqual(output.get(timeout=5), "ready")
                    with self.assertRaises(queue.Empty):
                        output.get(timeout=0.2)
                self.assertEqual(output.get(timeout=5), "acquired")
                process.wait(timeout=5)
                self.assertEqual(process.returncode, 0, process.stderr.read())
                reader.join(timeout=5)
            finally:
                if process is not None:
                    if process.poll() is None:
                        process.kill()
                    process.wait(timeout=5)
                    process.stdout.close()
                    process.stderr.close()


class OutputTransactionTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        for manager in (
            patch.dict(os.environ, {"OPENAI_API_BASE_URL": "https://api.example", "OPENAI_API_KEY": "fake-key"}, clear=True),
            patch.object(g, "load_environment", return_value=[]),
            patch.object(g.requests.sessions.Session, "request", side_effect=AssertionError("Network forbidden")),
        ):
            manager.start()
            self.addCleanup(manager.stop)

    def generate(self, session, output="original.png", manifest=None):
        options = ["generate", "--prompt", "audit", "--session", str(session),
                   "--output-dir", str(self.root), "--output", output]
        if manifest is not None:
            options += ["--manifest", str(manifest), "--asset-id", "card"]
        return g.run(parse_args(*options))

    def failed_transaction(self, manifest=None):
        session = self.root / "session.json"
        with patch.object(g, "call_openai", return_value=(PNG_BYTES, "image/png", "https://api.example")) as api:
            with patch.object(g, "write_image_exclusive", side_effect=OSError("simulated write failure")):
                with self.assertRaises((ValueError, g.ImageAPIError)) as caught:
                    self.generate(session, manifest=manifest)
            self.assertIn("recover --journal", str(caught.exception))
            api.assert_called_once()
        return session, g.pending_path(manifest or session)

    def test_same_explicit_output_is_checked_after_waiting_before_second_api_call(self):
        first_api_started = threading.Event()
        second_started = threading.Event()
        real_acquire = g.acquire_output_locks

        def acquire(stack, paths, dry_run):
            if threading.current_thread().name.endswith("_1"):
                second_started.set()
            return real_acquire(stack, paths, dry_run)

        def fake_api(*args, **kwargs):
            first_api_started.set()
            if not second_started.wait(timeout=5):
                raise AssertionError("Second task never reached output locking")
            return PNG_BYTES, "image/png", "https://api.example"

        def worker(index):
            try:
                return self.generate(self.root / f"session-{index}.json", "shared.png")
            except Exception as exc:
                return exc

        with patch.object(g, "acquire_output_locks", side_effect=acquire), patch.object(g, "call_openai", side_effect=fake_api) as api:
            with ThreadPoolExecutor(max_workers=2) as pool:
                first = pool.submit(worker, 0)
                self.assertTrue(first_api_started.wait(timeout=5))
                second = pool.submit(worker, 1)
                results = [first.result(timeout=10), second.result(timeout=10)]
            api.assert_called_once()
        self.assertIsInstance(results[0], dict)
        self.assertIsInstance(results[1], Exception)
        self.assertIn("拒绝覆盖", str(results[1]))
        self.assertEqual((self.root / "shared.png").read_bytes(), PNG_BYTES)
        self.assertEqual(list(self.root.glob("*.pending.json")), [])

    def test_output_lock_alias_cannot_truncate_reference_input(self):
        output = self.root / "target.png"
        source = output_lock_paths([output])[0]
        source.write_bytes(PNG_BYTES)
        with patch.object(g, "call_openai") as api:
            with self.assertRaises((ValueError, g.ImageAPIError)):
                g.run(parse_args("reference", "--prompt", "audit", "--image", str(source),
                                 "--output", str(output), "--session", str(self.root / "session.json")))
            api.assert_not_called()
        self.assertEqual(source.read_bytes(), PNG_BYTES)

    def test_session_lock_alias_cannot_truncate_reference_input(self):
        session = self.root / "session.json"
        source = Path(str(session) + ".lock")
        source.write_bytes(PNG_BYTES)
        with patch.object(g, "call_openai") as api:
            with self.assertRaises((ValueError, g.ImageAPIError)):
                g.run(parse_args("reference", "--prompt", "audit", "--image", str(source),
                                 "--session", str(session), "--output", str(self.root / "target.png")))
            api.assert_not_called()
        self.assertTrue(source.exists())
        self.assertEqual(source.read_bytes(), PNG_BYTES)

    def test_manifest_lock_alias_cannot_truncate_prompt_file(self):
        manifest = self.root / "manifest.json"
        prompt = Path(str(manifest) + ".lock")
        prompt.write_text("User prompt that must be kept", encoding="utf-8")
        with patch.object(g, "call_openai") as api:
            with self.assertRaises((ValueError, g.ImageAPIError)):
                g.run(parse_args("generate", "--prompt-file", str(prompt),
                                 "--manifest", str(manifest), "--asset-id", "card",
                                 "--output", str(self.root / "target.png")))
            api.assert_not_called()
        self.assertTrue(prompt.exists())
        self.assertEqual(prompt.read_text(encoding="utf-8"), "User prompt that must be kept")

    def test_recovery_to_new_output_preserves_collision_and_updates_both_metadata_files(self):
        manifest = self.root / "manifest.json"
        session, journal = self.failed_transaction(manifest)
        original = self.root / "original.png"
        original.write_bytes(b"existing user image")
        new_output = self.root / "recovered.png"
        with patch.object(g, "call_openai") as api:
            with self.assertRaises((ValueError, g.ImageAPIError)):
                g.run(parse_args("recover", "--journal", str(journal)))
            self.assertTrue(journal.exists())
            result = g.run(parse_args("recover", "--journal", str(journal), "--output", str(new_output)))
            api.assert_not_called()
        self.assertEqual(original.read_bytes(), b"existing user image")
        self.assertEqual(new_output.read_bytes(), PNG_BYTES)
        self.assertEqual(Path(result["image"]), new_output)
        stored_session = json.loads(session.read_text(encoding="utf-8"))
        asset = json.loads(manifest.read_text(encoding="utf-8"))["assets"]["card"]
        self.assertEqual(g.last_output(stored_session, session), new_output)
        self.assertEqual(g.manifest_file_path(manifest, asset["image"]), new_output)
        self.assertEqual(g.manifest_file_path(manifest, asset["session"]), session)
        self.assertEqual(len(stored_session["turns"]), 1)
        self.assertFalse(g.pending_path(session).exists())
        self.assertFalse(g.pending_path(manifest).exists())

    def test_recovery_to_new_output_updates_session_without_manifest(self):
        session, journal = self.failed_transaction()
        original = self.root / "original.png"
        original.write_bytes(b"existing user image")
        new_output = self.root / "session-only.png"
        with patch.object(g, "call_openai") as api:
            result = g.run(parse_args("recover", "--journal", str(journal), "--output", str(new_output)))
            api.assert_not_called()
        self.assertIsNone(result["manifest"])
        self.assertEqual(Path(result["image"]), new_output)
        self.assertEqual(new_output.read_bytes(), PNG_BYTES)
        self.assertEqual(original.read_bytes(), b"existing user image")
        stored = json.loads(session.read_text(encoding="utf-8"))
        self.assertEqual(g.last_output(stored, session), new_output)
        self.assertEqual(len(stored["turns"]), 1)
        self.assertFalse(journal.exists())

    def test_recovery_new_output_never_overwrites_another_file(self):
        session, journal = self.failed_transaction()
        occupied = self.root / "occupied.png"
        occupied.write_bytes(b"keep this file")
        before = journal.read_bytes()
        with patch.object(g, "call_openai") as api:
            with self.assertRaises((ValueError, OSError, g.ImageAPIError)):
                g.run(parse_args("recover", "--journal", str(journal), "--output", str(occupied)))
            api.assert_not_called()
        self.assertEqual(occupied.read_bytes(), b"keep this file")
        self.assertEqual(journal.read_bytes(), before)
        self.assertFalse(session.exists())

    def test_failed_redirected_recovery_can_resume_without_original_output_or_api(self):
        manifest = self.root / "manifest.json"
        session, journal = self.failed_transaction(manifest)
        original = self.root / "original.png"
        original.write_bytes(b"existing user image")
        new_output = self.root / "recovered.png"
        real_write = g.write_json_atomic

        def fail_session(path, payload):
            if path == session:
                raise PermissionError("session temporarily unavailable")
            return real_write(path, payload)

        with patch.object(g, "call_openai") as api:
            with patch.object(g, "write_json_atomic", side_effect=fail_session):
                for _ in range(2):
                    with self.assertRaises((ValueError, OSError, g.ImageAPIError)):
                        g.run(parse_args("recover", "--journal", str(journal), "--output", str(new_output)))
                    self.assertTrue(journal.exists())
                    self.assertTrue(g.pending_path(session).exists())
            # 新目标写入日志后，即使不再次传 --output 也能继续同一提交。
            result = g.run(parse_args("recover", "--journal", str(journal)))
            api.assert_not_called()
        self.assertEqual(Path(result["image"]), new_output)
        self.assertEqual(new_output.read_bytes(), PNG_BYTES)
        self.assertEqual(original.read_bytes(), b"existing user image")
        self.assertEqual(len(json.loads(session.read_text(encoding="utf-8"))["turns"]), 1)
        self.assertFalse(journal.exists())
        self.assertFalse(g.pending_path(session).exists())
        self.assertFalse(g.pending_path(manifest).exists())

    def test_recovery_from_old_session_journal_uses_new_manifest_journal(self):
        manifest = self.root / "manifest.json"
        session, manifest_journal = self.failed_transaction(manifest)
        session_journal = g.pending_path(session)
        original = self.root / "original.png"
        original.write_bytes(b"existing user image")
        new_output = self.root / "recovered.png"
        real_write = g.write_json_atomic

        def fail_secondary_journal(path, payload):
            if path == session_journal:
                raise PermissionError("secondary journal unavailable")
            return real_write(path, payload)

        with patch.object(g, "call_openai") as api:
            with patch.object(g, "write_json_atomic", side_effect=fail_secondary_journal):
                with self.assertRaises((ValueError, OSError, g.ImageAPIError)):
                    g.run(parse_args("recover", "--journal", str(manifest_journal), "--output", str(new_output)))
            self.assertEqual(json.loads(manifest_journal.read_text(encoding="utf-8"))["result"]["image"], str(new_output))
            self.assertEqual(json.loads(session_journal.read_text(encoding="utf-8"))["result"]["image"], str(original))
            self.assertFalse(new_output.exists())
            # 即使用户继续使用错误消息此前给出的旧副本，也不能恢复回旧目标。
            result = g.run(parse_args("recover", "--journal", str(session_journal)))
            api.assert_not_called()
        self.assertEqual(Path(result["image"]), new_output)
        self.assertEqual(new_output.read_bytes(), PNG_BYTES)
        self.assertEqual(original.read_bytes(), b"existing user image")
        self.assertEqual(g.last_output(json.loads(session.read_text(encoding="utf-8")), session), new_output)
        asset = json.loads(manifest.read_text(encoding="utf-8"))["assets"]["card"]
        self.assertEqual(g.manifest_file_path(manifest, asset["image"]), new_output)
        self.assertFalse(manifest_journal.exists())
        self.assertFalse(session_journal.exists())

    def test_recovery_output_lock_alias_cannot_truncate_recorded_reference(self):
        session = self.root / "session.json"
        original = self.root / "original.png"
        new_output = self.root / "recovered.png"
        reference = output_lock_paths([new_output])[0]
        reference.write_bytes(PNG_BYTES)
        with patch.object(g, "call_openai", return_value=(PNG_BYTES, "image/png", "https://api.example")):
            with patch.object(g, "write_image_exclusive", side_effect=OSError("simulated write failure")):
                with self.assertRaises((ValueError, g.ImageAPIError)):
                    g.run(parse_args("reference", "--prompt", "audit", "--image", str(reference),
                                     "--session", str(session), "--output", str(original)))
        journal = g.pending_path(session)
        before = journal.read_bytes()
        with patch.object(g, "call_openai") as api:
            with self.assertRaisesRegex((ValueError, g.ImageAPIError), "锁不能覆盖"):
                g.run(parse_args("recover", "--journal", str(journal), "--output", str(new_output)))
            api.assert_not_called()
        self.assertEqual(reference.read_bytes(), PNG_BYTES)
        self.assertEqual(journal.read_bytes(), before)
        self.assertFalse(new_output.exists())

    def test_invalid_other_asset_prompt_path_returns_json_without_mutating_journals(self):
        manifest = self.root / "manifest.json"
        session, journal = self.failed_transaction(manifest)
        journals = [journal, g.pending_path(session)]
        for path in journals:
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["manifest_payload"]["assets"]["unrelated"] = {"prompt_file": 7}
            path.write_text(json.dumps(payload), encoding="utf-8")
        before = {path: path.read_bytes() for path in journals}
        output = self.root / "recovered.png"
        stderr = io.StringIO()
        with patch.object(g, "call_openai") as api, patch.object(sys, "argv", [g.__file__, "recover", "--journal", str(journal), "--output", str(output)]), patch.object(sys, "stderr", stderr):
            self.assertEqual(g.main(), 1)
            api.assert_not_called()
        error = json.loads(stderr.getvalue())
        self.assertEqual(error["error_code"], "validation_error")
        self.assertEqual(error["stage"], "validation")
        self.assertIsNone(error["provider"])
        self.assertEqual({path: path.read_bytes() for path in journals}, before)
        self.assertFalse(output.exists())
        self.assertFalse(session.exists())

    def test_recovery_metadata_lock_alias_cannot_truncate_recorded_reference(self):
        manifest = self.root / "manifest.json"
        session, journal = self.failed_transaction(manifest)
        source = Path(str(manifest) + ".lock")
        source.write_bytes(PNG_BYTES)
        journals = [journal, g.pending_path(session)]
        for path in journals:
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["session_payload"]["turns"][-1]["input_images"] = [source.name]
            path.write_text(json.dumps(payload), encoding="utf-8")
        before = {path: path.read_bytes() for path in journals}
        output = self.root / "recovered.png"
        with patch.object(g, "call_openai") as api:
            with self.assertRaises((ValueError, g.ImageAPIError)):
                g.run(parse_args("recover", "--journal", str(journal), "--output", str(output)))
            api.assert_not_called()
        self.assertTrue(source.exists())
        self.assertEqual(source.read_bytes(), PNG_BYTES)
        self.assertEqual({path: path.read_bytes() for path in journals}, before)
        self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
