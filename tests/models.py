#!/usr/bin/env python3
"""Two real model sessions in one process; requires first-use BGE/Gemma downloads.
Run: python3 tests/models.py path/to/publish/memfind
"""
import json
from pathlib import Path
import subprocess
import sys
import tempfile

REPO = Path(__file__).resolve().parents[1]
app = [str(Path(sys.argv[1]).resolve())] if len(sys.argv) > 1 else ["dotnet", str(REPO / "bin/Release/net10.0/memfind.dll")]
with tempfile.TemporaryDirectory(prefix="memfind-models-") as directory:
    store = Path(directory)
    (store / ".memfind").mkdir()
    for suffix, name, dim in [("", "bge-small-en-v1.5-int8", 384), (".gemma", "embeddinggemma-300m-int8", 768)]:
        index = dict(model=name, dim=dim, builtAt="2026-10-22T00:00:00Z", items=[dict(file="fixture.md", line=i+1, theme="test", title="Fixture", description="A candidate with enough metadata to trigger collection during model preloading.", textHash=str(i), vec=[0.0123456789] * dim) for i in range(1024)])
        (store / f".memfind/index{suffix}.json").write_text(json.dumps(index))
    for model in ("bge", "gemma", "both"):
        result = subprocess.run(app + ["test", "--store", str(store), "--memories", "--model", model, "--json", "-k", "1"], capture_output=True, text=True, timeout=180)
        assert result.returncode == 0, (model, result.returncode, result.stderr)
        hits = [json.loads(line) for line in result.stdout.splitlines()]
        assert len(hits) == 1 and hits[0]["file"] == "fixture.md", hits
        if model == "both":
            assert {leg["model"] for leg in hits[0]["models"]} == {"bge", "gemma"}, hits
            assert all(leg["rank"] == 1 for leg in hits[0]["models"]), hits
            assert hits[0]["score"] == round(2 / 61, 4), hits
    print("PASS: BGE and Gemma alone and together, two-session inference and fused ranks")
