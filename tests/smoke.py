#!/usr/bin/env python3
"""End-to-end CLI checks with real bge embeddings and isolated fixtures.
Run after dotnet build -c Release: python3 tests/smoke.py
"""
import json
import os
from pathlib import Path
import subprocess
import shutil
import sys
import tempfile

REPO = Path(__file__).resolve().parents[1]
APP = [str(Path(sys.argv[1]).resolve())] if len(sys.argv) > 1 else ["dotnet", str(REPO / "bin/Release/net10.0/memfind.dll")]

with tempfile.TemporaryDirectory(prefix="memfind-smoke-") as directory:
    root = Path(directory)
    store, claude, codex, memories = [root / name for name in ("store", "claude", "codex", "memories")]
    for path in (store, claude, codex, memories):
        path.mkdir()
    base = ["--store", str(store), "--claude-projects", str(claude),
            "--codex-sessions", str(codex), "--codex-memories", str(memories), "--threads", "2"]

    def run(*args, code=0):
        result = subprocess.run(APP + base + list(args), capture_output=True, text=True, timeout=180)
        assert result.returncode == code, (args, result.returncode, result.stdout, result.stderr)
        return result

    def rows(*args):
        return [json.loads(line) for line in run(*args, "--json").stdout.splitlines()]

    def write(path, records):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(record) + "\n" for record in records))

    def claude_message(uuid, role, text):
        return dict(type=role, uuid=uuid, timestamp="2026-10-22T12:00:00Z", sessionId="claude-session",
                    cwd="/project/claude", message=dict(role=role, content=text))

    def meta(id, agent=False):
        return dict(type="session_meta", payload=dict(id=id, cwd="/project/codex",
                    source={"subagent": {"thread_spawn": {}}} if agent else "cli"))

    def message(role, text):
        return dict(type="response_item", timestamp="2026-10-22T12:00:00Z",
                    payload=dict(type="message", role=role, content=[dict(type="input_text" if role == "user" else "output_text", text=text)]))

    main = codex / "2026/10/22/rollout-main.jsonl"
    write(main, [meta("codex-session"), message("user", "How do I bake sourdough bread?"),
                 dict(type="event_msg", payload=dict(type="user_message", message="DUPLICATE_DO_NOT_INDEX")),
                 message("assistant", "Ferment the sourdough starter overnight, knead the bread dough and bake in a hot oven."),
                 message("developer", "SYSTEM_DO_NOT_INDEX"),
                 dict(type="response_item", payload=dict(type="function_call_output", output="TOOL_DO_NOT_INDEX")),
                 {"type": "response_item", "payload": {"type": "reasoning", "summary": "REASONING_DO_NOT_INDEX"}}])
    agent = codex / "2026/10/22/rollout-agent.jsonl.zst"
    shutil.copyfile(REPO / "tests/fixtures/rollout-agent.jsonl.zst", agent)
    write(claude / "project/session.jsonl", [claude_message("claude-user", "user", "Explain how to deploy a Docker container onto Kubernetes safely."),
          claude_message("claude-assistant", "assistant", "Deploy a container by creating a Kubernetes Deployment and Service. " * 4)])
    write(claude / "project/subagents/agent.jsonl", [claude_message("claude-agent", "user", "Check the Kubernetes deployment health and container readiness probes.")])
    (store / "MEMORY.md").write_text("- [Kubernetes rollout](kubernetes.md)\n")
    (store / "kubernetes.md").write_text("---\nname: Kubernetes rollout\ndescription: Deploy Docker containers with Kubernetes readiness probes.\n---\n")
    (memories / "MEMORY.md").write_text("# Bakery\nSourdough bread needs a mature starter and overnight fermentation.\n\n# Gardening\nTomato plants need sunshine and regular watering.\n")
    (memories / "memory_summary.md").write_text("# Preferences\nUse Kubernetes readiness probes during Docker deployments.\n")
    (memories / "rollout_summaries").mkdir()
    (memories / "rollout_summaries/baking.md").write_text("# Baking session\nThe best sourdough bread uses slow fermentation in a cool kitchen.\n")
    (memories / "raw_memories.md").write_text("RAW_DUPLICATE_DO_NOT_INDEX")

    run("--help")
    run("query", "--chats", "--memories", code=2)
    run("index")
    stats = rows("stats")[0]
    assert stats["chatItems"] == 6 and stats["transcripts"] == 4, stats
    assert stats["memories"] == 5, stats
    hits = rows("sourdough bread", "-k", "20")
    assert rows("sourdough bread", "-k", "20", "--recency-boost", "10", "--recency-half-life", "1") == hits
    assert {h["source"] for h in hits} == {"memory", "chat", "agent", "codex", "codex-agent"}, hits
    assert len([h for h in hits if h["source"] != "memory"]) == 6
    assert not any("DO_NOT_INDEX" in h.get("text", h.get("description", "")) for h in hits)
    bakery = rows("sourdough bread", "--memories", "-k", "1")[0]
    assert bakery["theme"] == "codex" and bakery["line"] > 0, bakery
    chat = next(h for h in hits if h["source"] == "codex" and h["role"] == "user")
    shown = rows("show", f"{chat['file']}:{chat['line']}", "-c", "1")
    assert len(shown) == 2 and shown[0]["anchor"] and "sourdough" in shown[0]["text"], shown
    assert shown[0]["uuid"] == chat["uuid"]
    compressed = next(h for h in hits if h["source"] == "codex-agent")
    assert rows("show", f"{compressed['file']}:{compressed['line']}", "-c", "0")[0]["text"] == compressed["text"]
    run("show", f"{main}:3", code=1)
    assert "read 0 file(s)" in run("index").stderr
    assert rows("stats")[0]["chatItems"] == 6
    with main.open("a") as output:
        output.write(json.dumps(message("user", "What temperature should the sourdough oven reach?")))
    run("index")
    assert rows("stats")[0]["chatItems"] == 6
    with main.open("a") as output:
        output.write("\n")
    run("index")
    assert rows("stats")[0]["chatItems"] == 7
    run("index", "--no-agents")
    assert rows("stats")[0]["chatItems"] == 5
    assert {h["source"] for h in rows("bread", "--chats", "-k", "20")} == {"chat", "codex"}
    run("index")
    assert rows("stats")[0]["chatItems"] == 7
    shutil.copyfile(REPO / "tests/fixtures/rollout-agent-updated.jsonl.zst", agent)
    run("index")
    assert rows("stats")[0]["chatItems"] == 8
    agent.unlink()
    run("index")
    assert rows("stats")[0]["chatItems"] == 6
    memory_file = memories / "MEMORY.md"
    memory_file.write_text("# Bakery\nSourdough now ferments for forty eight hours.\n")
    run("index", "--memory-only")
    assert rows("stats")[0]["memories"] == 4
    # A default query remains useful when only one source has an index.
    index_path = store / ".memfind/index.json"
    memory_index = index_path.read_bytes()
    index_path.unlink()
    assert all(h["source"] != "memory" for h in rows("bread"))
    run("bread", "--memories", code=1)
    index_path.write_bytes(memory_index)
    vec = store / ".memfind/chats.vec"
    vec.rename(vec.with_suffix(".backup"))
    assert all(h["source"] == "memory" for h in rows("bread"))
    run("bread", "--chats", code=1)
    vec.with_suffix(".backup").rename(vec)
    # Old metadata must be refused, then rebuilt with source tags and absolute paths.
    state_path = store / ".memfind/chats.state.json"
    state = json.loads(state_path.read_text())
    state["metaVersion"] = 2
    state_path.write_text(json.dumps(state))
    run("bread", "--chats", code=1)
    run("index")
    assert rows("stats")[0]["metaVersion"] == 3
    # --sources picks which transcript roots are indexed; Codex memory counts as Codex.
    other = ["--store", str(root / "sources-store")]
    (root / "sources-store").mkdir()
    (root / "sources-store/MEMORY.md").write_text("- [Kubernetes rollout](kubernetes.md)\n")
    shutil.copyfile(store / "kubernetes.md", root / "sources-store/kubernetes.md")
    assert "rebuilding" not in run("index", *other).stderr
    stats = rows("stats", *other)[0]
    assert stats["chatItems"] == 6 and stats["memories"] == 4, stats
    assert "sources=All" in run("index", "--sources", "claude", *other).stderr
    stats = rows("stats", *other)[0]
    assert stats["chatItems"] == 3 and stats["transcripts"] == 2 and stats["memories"] == 1, stats
    assert {h["source"] for h in rows("bread", "--chats", "-k", "20", *other)} <= {"chat", "agent"}
    assert all(h["theme"] != "codex" for h in rows("sourdough bread", "--memories", "-k", "20", *other))
    stderr = run("index", "--sources", "claude", *other).stderr
    assert "rebuilding" not in stderr and "read 0 file(s)" in stderr, stderr
    run("stats", "--sources", "claude")
    assert "sources=Claude " in run("index", "--sources", "claude,codex", *other).stderr
    stats = rows("stats", *other)[0]
    assert stats["chatItems"] == 6 and stats["memories"] == 4, stats
    assert "rebuilding" in run("index", "--sources", "codex", *other).stderr
    stats = rows("stats", *other)[0]
    assert stats["chatItems"] == 3 and stats["memories"] == 4, stats
    assert {h["source"] for h in rows("bread", "--chats", "-k", "20", *other)} <= {"codex", "codex-agent"}
    # An index state written before --sources existed means both, and is not rebuilt.
    run("index", *other)
    state_path = root / "sources-store/.memfind/chats.state.json"
    state = json.loads(state_path.read_text())
    del state["sources"]
    state_path.write_text(json.dumps(state))
    assert "rebuilding" not in run("index", *other).stderr
    assert rows("stats", *other)[0]["chatItems"] == 6
    for bad in ("foo", "", "claude,", "claude,foo"):
        run("index", "--sources", bad, *other, code=2)
    run("index", "--sources", *other, code=2)
    run("index", "--sources", "claude", "--claude-projects", str(root / "absent"), *other, code=1)
    fused = rows("bread", "--model", "both", "-k", "20")
    assert all(h["models"][0]["model"] == "bge" for h in fused)
    missing = root / "missing"
    run("query", "--codex-sessions", str(missing), "--claude-projects", str(missing), "--chats")
    # Check environment-based defaults without touching actual user histories.
    env = os.environ.copy()
    env["MEMFIND_STORE"] = str(root / "env-store")
    env["CODEX_HOME"] = str(root / "custom-codex")
    shutil.copytree(memories, root / "custom-codex/memories")
    result = subprocess.run(APP + ["index", "--memory-only", "--threads", "2"], env=env, capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stderr
    result = subprocess.run(APP + ["stats", "--json"], env=env, capture_output=True, text=True, check=True)
    stats = json.loads(result.stdout)
    assert stats["store"] == env["MEMFIND_STORE"] and stats["memories"] == 3, stats
    tilde = "~/" + root.name + "-unused"
    result = subprocess.run(APP + ["stats", "--json", "--store", tilde], env=env, capture_output=True, text=True, check=True)
    assert json.loads(result.stdout)["store"] == str(Path.home() / (root.name + "-unused"))
    print("PASS: real embeddings; Claude/Codex/agents, including zstd and archive replacement; Codex memory chunks and anchors; show; filters; incremental/partial append; deletion; memory edits; old-index migration; source/model fallback; path flags/env/tilde")
