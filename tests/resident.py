#!/usr/bin/env python3
"""Real-model resident regression. Pass a published executable; models must already be cached."""
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

if len(sys.argv) != 2:
    raise SystemExit("Usage: python3 tests/resident.py path/to/publish/memfind")

original = Path(sys.argv[1]).resolve()
with tempfile.TemporaryDirectory(prefix="memfind-resident-") as directory:
    root = Path(directory)
    # Isolate the host tag and exercise executable paths containing spaces and shell quotes.
    publish = root / "bin with spaces 'quoted'"
    shutil.copytree(original.parent, publish, ignore=shutil.ignore_patterns("*.dSYM", "*.pdb"))
    app = publish / original.name
    store = root / "store"
    store.mkdir()
    query = "Use a store-local file lock to prevent simultaneous indexing jobs."
    (store / "lock.md").write_text(f"---\nname: Index lock\ndescription: {query}\n---\n", encoding="utf-8")
    log = root / "hook.log"
    env = dict(os.environ, MEMFIND_HOOK_LOG=str(log))
    command = ["dotnet", str(app)] if app.suffix == ".dll" else [str(app)]
    base = command + ["--store", str(store), "--threads", "1", "--offline"]
    event = json.dumps({"hook_event_name": "UserPromptSubmit", "prompt": query})
    children = set()

    def hook():
        started = time.monotonic()
        result = subprocess.run(base + ["context-hook"], input=event, capture_output=True,
                                text=True, env=env, timeout=12)
        # A child retaining stdout makes communicate() time out, even after the hook exits.
        lines = log.read_text().splitlines() if log.exists() else []
        children.update(int(pid) for pid in re.findall(r"started\(pid=(\d+),", "\n".join(lines)))
        assert result.returncode == 0 and not result.stderr, result
        context = json.loads(result.stdout)["hookSpecificOutput"]["additionalContext"]
        assert "lock.md" in context and len(context) <= 3000, context
        print(f"{time.monotonic() - started:.3f}s {lines[-1]}", flush=True)
        return context, lines[-1]

    try:
        subprocess.run(base + ["index", "--model", "both", "--memory-only", "--sources", "claude"],
                       capture_output=True, text=True, timeout=180, check=True)
        first, line = hook()
        assert "gemma=started(pid=" in line, "first hook must start its own resident"
        deadline = time.monotonic() + 20
        while True:
            context, line = hook()
            if "gemma=host" in line:
                break
            assert time.monotonic() < deadline, "later hooks never reached the resident"
            time.sleep(.1)
        assert context == first, "resident and local inference produce the same context"
        duplicate = subprocess.run(base + ["serve"], capture_output=True, text=True, timeout=3)
        assert duplicate.returncode == 0 and not duplicate.stdout and not duplicate.stderr, duplicate
        again, line = hook()
        assert "gemma=host" in line and again == first, "resident survives its starting hook"
        assert len(children) == 1, "warm hooks must reuse the resident, not spawn more processes"
        print("PASS: detached autostart, stdout EOF, quoted paths, one resident and equal retrieval", flush=True)
    finally:
        if log.exists():
            children.update(int(pid) for pid in re.findall(r"started\(pid=(\d+),", log.read_text()))
        for pid in children:
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        # Windows cannot remove an executable until its resident has exited.
        deadline = time.monotonic() + 5
        while children and time.monotonic() < deadline:
            try:
                shutil.rmtree(publish)
                break
            except PermissionError:
                time.sleep(.1)
