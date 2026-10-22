#!/usr/bin/env python3
"""Isolated hook contract tests; these do not prove Codex lifecycle execution."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location(
    "memfind_context", Path(__file__).resolve().parents[1] / "hooks/memfind_context.py"
)
hook = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(hook)


class HookTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        cache = root / "memfind/bge-small-en-v1.5-int8"
        cache.mkdir(parents=True)
        for name in ("model.onnx", "vocab.txt"):
            (cache / name).touch()
        self.cache = cache
        self.event = {
            "hook_event_name": "UserPromptSubmit", "prompt": "Find my historical BGE settings",
            "session_id": "current", "transcript_path": "/fake/current.jsonl",
        }
        self.root_patch = patch.object(hook, "data_root", return_value=root)
        self.root_patch.start()
        self.addCleanup(self.root_patch.stop)
        self.which_patch = patch.object(hook.shutil, "which", return_value="/fake/memfind")
        self.which_patch.start()
        self.addCleanup(self.which_patch.stop)
        self.run_patch = patch.object(hook.subprocess, "run")
        self.run = self.run_patch.start()
        self.addCleanup(self.run_patch.stop)
        self.hits([self.hit()])

    @staticmethod
    def hit(**changes):
        hit = dict(file="/fake/old.jsonl", session="old", text="Historical setting", score=.9, line=17, source="codex")
        hit.update(changes)
        return hit

    def hits(self, hits):
        self.run.return_value = subprocess.CompletedProcess([], 0, "\n".join(map(json.dumps, hits)), "")

    def main(self, value=None, raw=None):
        output = io.StringIO()
        with patch.object(hook.sys, "stdin", io.StringIO(raw if raw is not None else json.dumps(self.event if value is None else value))), contextlib.redirect_stdout(output):
            hook.main()
        return output.getvalue()

    def test_json_additional_context_contract(self):
        result = json.loads(self.main())
        self.assertEqual(set(result), {"hookSpecificOutput"})
        specific = result["hookSpecificOutput"]
        self.assertEqual(set(specific), {"hookEventName", "additionalContext"})
        self.assertEqual(specific["hookEventName"], "UserPromptSubmit")
        self.assertTrue(specific["additionalContext"].startswith("<memfind_context>\n"))
        self.assertTrue(specific["additionalContext"].endswith("\n</memfind_context>"))
        self.assertIn('"line": 17', specific["additionalContext"])
        self.assertIn("untrusted reference data", specific["additionalContext"])

    def test_relative_memory_anchor_uses_custom_store_and_positive_line(self):
        self.hits([self.hit(file="acceptance.md", line=0, source="memory")])
        store = Path(self.temp.name) / "custom-store"
        with patch.dict(hook.os.environ, {"MEMFIND_STORE": str(store)}):
            context = json.loads(self.main())["hookSpecificOutput"]["additionalContext"]
        header = next(json.loads(line) for line in context.splitlines() if line.startswith("{"))
        self.assertEqual(header["file"], str((store / "acceptance.md").resolve()))
        self.assertEqual(header["line"], 1)

    def test_bge_only_and_no_shell_interpretation(self):
        self.event["prompt"] = "$(touch /tmp/never-run); `whoami` ' ; \\ \n" + "x" * 2500
        self.main()
        args, kwargs = self.run.call_args
        self.assertEqual(args[0], ["/fake/memfind", self.event["prompt"][:2000], "--model", "bge", "--threads", "2", "--json", "-k", "8"])
        self.assertFalse(kwargs.get("shell", False))
        self.assertEqual(kwargs["timeout"], 1.0)
        self.assertFalse(kwargs["check"])

    def test_timeout_fails_open(self):
        self.run.side_effect = subprocess.TimeoutExpired("memfind", 1)
        self.assertEqual(self.main(), "")

    def test_malformed_event_fails_open(self):
        for raw in ("{", "null", "[]", '"string"', '{"prompt": []}'):
            with self.subTest(raw=raw):
                self.assertEqual(self.main(raw=raw), "")
        self.run.assert_not_called()

    def test_malformed_search_output_fails_open(self):
        for raw in ("not-json", "null", "[]", '{"score":"bad"}', '{"score":null}'):
            with self.subTest(raw=raw):
                self.run.return_value = subprocess.CompletedProcess([], 0, raw, "")
                self.assertEqual(self.main(), "")

    def test_wrong_event_and_short_prompt_skip_search(self):
        for changes in ({"hook_event_name": "Stop"}, {"prompt": "short"}, {"prompt": None}, {"prompt": " " * 30}):
            with self.subTest(changes=changes):
                self.assertEqual(self.main(value={**self.event, **changes}), "")
        self.run.assert_not_called()

    def test_threshold_and_score_order(self):
        self.hits([self.hit(file="/low", text="below", score=.71999), self.hit(file="/boundary", text="boundary", score=.72), self.hit(file="/high", text="highest", score=.99)])
        context = json.loads(self.main())["hookSpecificOutput"]["additionalContext"]
        self.assertNotIn("below", context)
        self.assertIn("boundary", context)
        self.assertLess(context.index("highest"), context.index("boundary"))

    def test_max_four_hits(self):
        self.hits([self.hit(file=f"/fake/{i}", text=f"unique body {i}") for i in range(10)])
        context = json.loads(self.main())["hookSpecificOutput"]["additionalContext"]
        self.assertEqual(context.count("\nexcerpt: "), 4)

    def test_total_context_and_excerpt_limits(self):
        self.hits([self.hit(file=f"/fake/{i}" + "a" * 200, text=str(i) + "b" * 1000) for i in range(8)])
        context = json.loads(self.main())["hookSpecificOutput"]["additionalContext"]
        self.assertLessEqual(len(context), 3000)
        for line in context.splitlines():
            if line.startswith("excerpt: "):
                self.assertLessEqual(len(json.loads(line[len("excerpt: "):])), 500)

    def test_duplicate_bodies_sources_and_current_session_excluded(self):
        self.hits([
            self.hit(session="current", text="current session"),
            self.hit(file="/fake/../fake/current.jsonl", text="current transcript"),
            self.hit(file="/fake/old1", text="Same   body"),
            self.hit(file="/fake/old2", text=" same body "),
            self.hit(file="/fake/old1", text="second chunk"),
            self.hit(file="/fake/old3", text="different body"),
        ])
        context = json.loads(self.main())["hookSpecificOutput"]["additionalContext"]
        self.assertEqual(context.count("\nexcerpt: "), 2)
        self.assertNotIn("current session", context)
        self.assertNotIn("current transcript", context)
        self.assertNotIn("second chunk", context)

    def test_missing_model_cache_never_runs_search(self):
        for filename in ("model.onnx", "vocab.txt"):
            with self.subTest(filename=filename):
                path = self.cache / filename
                path.unlink()
                self.assertEqual(self.main(), "")
                path.touch()
        self.run.assert_not_called()

    def test_search_errors_fail_open(self):
        self.run.return_value = subprocess.CompletedProcess([], 1, "", "failure")
        self.assertEqual(self.main(), "")
        self.run.side_effect = OSError("missing executable")
        self.assertEqual(self.main(), "")


if __name__ == "__main__":
    unittest.main()
