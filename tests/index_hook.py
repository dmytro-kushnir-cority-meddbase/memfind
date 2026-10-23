#!/usr/bin/env python3
"""Published-binary lifecycle test, real Gemma and isolated sources; never downloads a model."""
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time

original = Path(sys.argv[1]).resolve()
with tempfile.TemporaryDirectory(prefix="memfind-index-hook-") as directory:
    root = Path(directory)
    publish = root / "bin with spaces 'quoted'"
    shutil.copytree(original.parent, publish, ignore=shutil.ignore_patterns("*.dSYM", "*.pdb"))
    app = publish / original.name
    command = ["dotnet", str(app)] if app.suffix == ".dll" else [str(app)]
    store, sessions, memories = [root / name for name in ("store", "sessions", "memories")]
    for path in (store, sessions, memories):
        path.mkdir()
    text = "Индексация переписки использует Gemma и файловую блокировку."
    (store / "gemma.md").write_text(f"---\nname: Gemma\ndescription: {text}\n---\n", encoding="utf-8")
    (memories / "MEMORY.md").write_text("# Русский чат\n" + text, encoding="utf-8")
    transcript = sessions / "rollout.jsonl"
    transcript.write_text(json.dumps({"type": "session_meta", "payload": {"id": "fixture", "cwd": str(root), "source": "cli"}}) + "\n", encoding="utf-8")

    def append_message():
        with transcript.open("a", encoding="utf-8") as output:
            output.write(json.dumps({"type": "response_item", "timestamp": "2026-10-23T12:00:00Z", "payload": {
                "type": "message", "role": "user", "content": [{"type": "input_text", "text": text}]}}) + "\n")

    append_message()
    env = dict(os.environ, MEMFIND_MODELS="gemma")
    base = command + ["--store", str(store), "--sources", "codex", "--codex-sessions", str(sessions),
                      "--codex-memories", str(memories), "--claude-projects", str(root / "empty"), "--threads", "1", "--offline"]

    def run(*args, input=None, code=0):
        result = subprocess.run(base + list(args), input=input, capture_output=True, text=True, env=env, timeout=15)
        assert result.returncode == code, (args, result)
        return result

    def request(source):
        before = time.monotonic()
        result = run("index-hook", input=json.dumps({"hook_event_name": "SessionStart", "source": source}))
        assert not result.stdout and not result.stderr, result
        assert time.monotonic() - before < 5, "hook launcher waited for indexing"

    def wait_for(memories, messages):
        deadline = time.monotonic() + 45
        while time.monotonic() < deadline:
            result = run("stats", "--json")
            stats = json.loads(result.stdout)
            if stats["memories"] == memories and stats["chatItems"] == messages and not list((store / ".memfind").glob("*.requested")):
                assert [model["key"] for model in stats["models"]] == ["gemma"], stats
                return
            time.sleep(.2)
        logs = "\n".join(path.read_text() for path in (store / ".memfind").glob("index-*.log"))
        raise AssertionError(("background indexing did not finish", stats, logs))

    try:
        request("startup")
        for _ in range(5):
            request("resume")
        wait_for(2, 1)
        hits = [json.loads(line) for line in run(text, "--json").stdout.splitlines()]
        assert any(hit["source"] == "codex" for hit in hits), hits
        assert not (store / ".memfind/index.json").exists(), "BGE memory index was built"
        assert not (store / ".memfind/chats.vec").exists(), "BGE chat index was built"
        for args in [(text, "--model", "bge"), ("index", "--model", "bge")]:
            assert "disabled" in run(*args, code=2).stderr

        (store / "another.md").write_text(f"---\nname: Ещё память\ndescription: {text}\n---\n", encoding="utf-8")
        append_message()
        request("compact")
        wait_for(3, 2)
        print("PASS: startup/compact indexing, detached launcher, incremental updates, Russian search and BGE disabled")
    finally:
        for path in (store / ".memfind").glob("index-*.log"):
            for pid in set(map(int, re.findall(r"worker pid=(\d+) started", path.read_text()))):
                try:
                    os.kill(pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            try:
                shutil.rmtree(publish)
                break
            except PermissionError:
                time.sleep(.1)
