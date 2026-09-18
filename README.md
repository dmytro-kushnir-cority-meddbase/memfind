# memfind

Semantic search over the Claude Code memory store (`C:\Git\docs\memory`) and the chat transcripts
under `%USERPROFILE%\.claude\projects`. One NativeAOT exe, no service, no server.

## Commands

| Command | What it does |
| --- | --- |
| `memfind index [options]` | Build both indexes into `<store>\.memfind\` |
| `memfind <query words...>` | Search memories |
| `memfind <query words...> --chats` | Search chat transcripts |
| `memfind <query words...> --all` | Search both (memories first) |
| `memfind show <file>:<line> [-c N]` | Print that transcript message in full, with neighbours |
| `memfind stats` | What is indexed |
| `memfind --help` | Usage |

## Flags

| Flag | Applies to | Meaning |
| --- | --- | --- |
| `--store <path>` | all | Memory store root (default `C:\Git\docs\memory`) |
| `-k N` | queries | Hits per source (default 5) |
| `-c N` | `show` | Kept messages of context on each side (default 3) |
| `--json` | queries, `show`, `stats` | JSONL instead of columns |
| `--chats` / `--all` | queries | Which index to search |
| `--chats-only` / `--memory-only` | `index` | Skip the other half |
| `--rebuild` | `index` | Ignore the incremental state, re-read every transcript |
| `--include-agents` | `index`, `show` | Index/keep subagent (sidechain) messages |
| `--fp32` | `index` | Use the fp32 model instead of int8 |
| `--threads N` | all | Intra-op and parse threads (default: half the cores) |

## Exit codes

| Code | When |
| --- | --- |
| 0 | Success |
| 1 | Missing or outdated index, missing transcript, or an anchor line that is not a kept message |
| 2 | Bad arguments |

## Output — text

Memory hit, two lines:

```
0.611  workstation   feedback_unsigned_installers.md   Unsigned installers
       He cancels the UAC prompt on anything unsigned ...
```

Chat hit, three lines — score row, anchor, preview:

```
0.628  2026-09-15 08:51  assistant   c--Git   755e4845
       C:\Users\me\.claude\projects\C--Git\755e4845-....jsonl:8412
       the 160-char preview, newlines collapsed ...
```

The second line is the anchor: feed it straight to `memfind show`.

`show`, one header plus the full text per message, blank line between; the anchored message's
header starts with `>>>` instead of `---`:

```
--- user  2026-09-15 08:50  line 8402
...full text...

>>> assistant  2026-09-15 08:51  line 8412
...full text...
```

## Output — `--json`

One object per line (JSONL). `ts` is the raw transcript timestamp (ISO-8601 UTC); text mode renders
it as local time. `score` is rounded to 4 decimals.

| Shape | Fields |
| --- | --- |
| Memory hit | `source="memory"`, `score`, `file`, `theme`, `title`, `description` |
| Chat hit | `source="chat"`, `score`, `file` (full transcript path), `line`, `uuid`, `chunk`, `session`, `cwd`, `role`, `ts`, `text` (the stored preview) |
| `show` | `role`, `ts`, `line`, `uuid`, `anchor` (bool), `text` (full) |
| `stats` | `store`, `memories`, `memoryBytes`, `memoryModel`, `chatItems`, `transcripts`, `chatModel`, `includeAgents`, `metaVersion`, `vecBytes`, `metaBytes`, `stateBytes`, `threads`, `batch`, `cores` |

`line` is the 1-based line of the message's JSONL record in `file`; `uuid` is that record's
top-level `uuid`. A long message is split into chunks that share `line` and `uuid` and differ by
`chunk` (0-based).

## On disk

All under `<store>\.memfind\` (with a `.gitignore` of `*`).

| File | Schema |
| --- | --- |
| `index.json` | `{model, dim, builtAt, items:[{file, theme, title, description, vec[384]}]}` — the memory index, read whole |
| `chats.vec` | Raw little-endian `float32[384]` per item, back to back; item *i* is at `i * 1536` bytes. Memory-mapped at search time |
| `chats.meta.jsonl` | One object per item, in `chats.vec` order: `{i, file, session, cwd, role, ts, line, uuid, chunk, text}`. `file` is relative to the projects root; `text` is the 300-char preview |
| `chats.state.json` | `{model, dim, metaVersion, includeAgents, files:[{path, length, lines, lastWriteUtc, itemsFrom, itemsCount}]}` — the incremental state: bytes and lines already consumed per transcript, and the item range they produced |

`metaVersion` is the layout of `chats.meta.jsonl` (currently **2**). A search against an older one
fails with `chats.meta.jsonl is old — run: memfind index --rebuild` and exit 1; `index` rebuilds
from scratch on the mismatch instead.

## Model

bge-small-en-v1.5, 384-dim, int8 by default (`--fp32` for the float model). Cached in
`%LOCALAPPDATA%\memfind\<model>\` as `model.onnx` + `vocab.txt`, downloaded from huggingface.co on
first use. `show` never loads it.

## SessionStart hook

`~/.claude/settings.json` runs the index on every session start, async so it never blocks:

```json
{ "type": "command", "command": "C:\\Users\\dkushnir\\bin\\memfind\\memfind.exe",
  "args": ["index"], "async": true, "timeout": 300 }
```

## The mutex

`index` takes `Global\memfind-index` without waiting. A second one — parallel sessions starting
together — prints `memfind: another index is running, skipping` and exits 0, so concurrent hooks
never race on `chats.vec`.
