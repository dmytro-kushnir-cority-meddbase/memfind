# memfind

Semantic search over the Claude Code memory store (`C:\Git\docs\memory`) and the chat transcripts
under `%USERPROFILE%\.claude\projects`. One NativeAOT exe, no service, no server.

## Commands

| Command | What it does |
| --- | --- |
| `memfind index [options]` | Build both indexes into `<store>\.memfind\` |
| `memfind <query words...>` | Search memories, chats and agent chats (memories first) |
| `memfind <query words...> --chats` | Search chat transcripts only (main and agent) |
| `memfind <query words...> --memories` | Search memories only |
| `memfind show <file>:<line> [-c N]` | Print that transcript message in full, with neighbours |
| `memfind stats` | What is indexed |
| `memfind --help` | Usage |

Every command takes `--model bge|bge-fp32|gemma|both`; `bge` is the default. On `index`, `both`
builds each model in turn; on a query it searches with each and fuses the two rankings.

## Flags

| Flag | Applies to | Meaning |
| --- | --- | --- |
| `--store <path>` | all | Memory store root (default `C:\Git\docs\memory`) |
| `-k N` | queries | Hits per source (default 5) |
| `-c N` | `show` | Kept messages of context on each side (default 3) |
| `--json` | queries, `show`, `stats` | JSONL instead of columns |
| `--chats` / `--chats-only` | queries | Search chat transcripts only (main and agent) |
| `--memories` / `--memory-only` | queries | Search memories only; with `--chats` it is an error (exit 2) |
| `--all` | queries | Search everything — the default, accepted for old callers |
| `--chats-only` / `--memory-only` | `index` | Skip the other half |
| `--rebuild` | `index` | Ignore the incremental state, re-read every transcript |
| `--no-agents` | `index` | Skip subagent transcripts and sidechain messages (indexed by default; `--include-agents` is accepted and does nothing) |
| `--model <name>` | `index`, queries, `stats` | `bge` (default), `bge-fp32`, `gemma`, or `both` |
| `--fp32` | `index` | Alias for `--model bge-fp32`; with `--model both` it is an error (exit 2) |
| `--gpu` | `index` | Try the CUDA execution provider, fall back to CPU |
| `--threads N` | all | Intra-op and parse threads (default: half the cores; a search uses 4 intra-op) |

## Exit codes

| Code | When |
| --- | --- |
| 0 | Success |
| 1 | Missing or outdated index, missing transcript, or an anchor line that is not a kept message |
| 2 | Bad arguments |

## Output — text

The first line of every hit starts with the score, then the source padded to six characters:
`memory` (memory store), `chat` (a main session transcript) or `agent` (a subagent transcript,
any path with a `subagents` directory, workflow agents included). A default search prints the
memory hits, a blank line, then the chat and agent hits ranked together.

Memory hit, two lines:

```
0.611  memory  workstation   feedback_unsigned_installers.md   Unsigned installers
       He cancels the UAC prompt on anything unsigned ...
```

Chat hit, three lines — score row, anchor, preview:

```
0.628  chat    2026-09-15 08:51  assistant   c--Git   755e4845
       C:\Users\me\.claude\projects\C--Git\755e4845-....jsonl:8412
       the 160-char preview, newlines collapsed ...
```

The second line is the anchor: feed it straight to `memfind show`.

Under `--model both` the score column is the fused score and the first line ends with the legs
that produced it: `[bge #1 0.780 gemma #1 0.592]`.

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
| Chat hit | `source="chat"` (main transcript) or `"agent"` (subagent transcript), `score`, `file` (full transcript path), `line`, `uuid`, `chunk`, `session`, `cwd`, `role`, `ts`, `text` (the stored preview) |
| `show` | `role`, `ts`, `line`, `uuid`, `anchor` (bool), `text` (full) |
| `stats` | `store`, `memories`, `memoryBytes`, `memoryModel`, `chatItems`, `transcripts`, `chatModel`, `includeAgents`, `metaVersion`, `vecBytes`, `metaBytes`, `stateBytes`, `threads`, `batch`, `cores`, `models` |

Under `--model both` a hit's `score` is the fused score and it carries `models`, one object per
contributing leg: `{model, score, rank}` — that leg's own cosine and 1-based rank. `stats` always
carries `models`, one object per model with an index on disk: `{key, memories, memoryBytes,
memoryModel, chatItems, transcripts, chatModel, vecBytes}`; the top-level fields stay bge's.

`line` is the 1-based line of the message's JSONL record in `file`; `uuid` is that record's
top-level `uuid`. A long message is split into chunks that share `line` and `uuid` and differ by
`chunk` (0-based).

## On disk

All under `<store>\.memfind\` (with a `.gitignore` of `*`).

| File | Schema |
| --- | --- |
| `index.json` | `{model, dim, builtAt, items:[{file, theme, title, description, vec[dim]}]}` — the memory index, read whole |
| `chats.vec` | Raw little-endian `float32[dim]` per item, back to back; item *i* is at `i * dim * 4` bytes. Memory-mapped at search time |
| `chats.meta.jsonl` | One object per item, in `chats.vec` order: `{i, file, session, cwd, role, ts, line, uuid, chunk, text}`. `file` is relative to the projects root; `text` is the 300-char preview |
| `chats.state.json` | `{model, dim, metaVersion, includeAgents, files:[{path, length, lines, lastWriteUtc, itemsFrom, itemsCount}]}` — the incremental state: bytes and lines already consumed per transcript, and the item range they produced |

Each model keeps its own set. bge (and `bge-fp32`, which shares them) uses the names above;
gemma inserts its key: `index.gemma.json`, `chats.gemma.vec`, `chats.gemma.meta.jsonl`,
`chats.gemma.state.json`. `dim` is 384 for bge, 768 for gemma.

`metaVersion` is the layout of `chats.meta.jsonl` (currently **2**). A search against an older one
fails with `chats.meta.jsonl is old — run: memfind index --rebuild` and exit 1; `index` rebuilds
from scratch on the mismatch instead.

## Models

| Key | Model | Dim | Files in `%LOCALAPPDATA%\memfind\<model>\` |
| --- | --- | --- | --- |
| `bge` (default) | bge-small-en-v1.5, int8 | 384 | `model.onnx`, `vocab.txt` |
| `bge-fp32` | bge-small-en-v1.5, fp32 | 384 | `model.onnx`, `vocab.txt` |
| `gemma` | embeddinggemma-300m, int8 | 768 | `model_quantized.onnx`, `model_quantized.onnx_data`, `tokenizer.model` |

Downloaded from huggingface.co on first use; `show` never loads a model. gemma's weights are
external to the graph, so both `.onnx` files must keep their published names and sit together.
Each model prefixes what it embeds — bge only its query, gemma both (`task: search result |
query: ` and `title: none | text: `) — and 512/256 token caps are shared, so the chat chunking
means the same thing under either. A bge search saves the graph ONNX Runtime optimises as
`model.optimized.onnx` beside the model and reuses it while `model.optimized.key` (model size,
mtime, runtime build) still matches; memfind-rs shares both files. gemma is never cached.

### `--model both`

Each leg is searched separately and the two rankings are fused with RRF,
`sum over legs of 1 / (60 + rank)`, taking 3k from each leg and cutting to k. The models' cosines
are on different scales (measured on one probe set: bge 0.57/0.80/0.74/0.69 vs gemma
0.23/0.56/0.44/0.44), so nothing but the ranks is comparable. A leg with no index on disk prints
one line to stderr and is skipped; with no legs left the query exits 1.

### `--gpu`

`index --gpu` runs the CUDA execution provider. The package is `Microsoft.ML.OnnxRuntime.Gpu`,
which costs 178 MB of `onnxruntime_providers_cuda.dll` in the publish output (245 MB total, against
23 MB for the CPU-only build).

cuDNN 9 ships separately from that package and a missing one surfaces as `NOT_IMPLEMENTED` on the
first `Run`, too late to fall back — so both directories are located up front and put on `PATH`,
and the provider is only appended when they are found. Searched, in order: `%MEMFIND_CUDNN%` /
`%MEMFIND_CUDA%`, then `%LOCALAPPDATA%\memfind\cudnn` and `C:\Tools` for `cudnn64_9.dll`, then
`C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA` for `cudart64_13.dll`. Not found prints
`cuda unavailable (...), using cpu`. cuDNN comes from
<https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/windows-x86_64/> — take the
`_cuda13` archive, since ORT 1.30's provider imports `cublas64_13.dll`, and only its `bin\*.dll`
are needed.

Measured here (RTX 5060, cuDNN 9.26, 256-token chunks, batch 64): gemma int8 13.2 items/s on CPU
vs 182.5 on CUDA; bge int8 73.8 on CPU. The gemma memory index takes 29.9 s on CPU and 4.7 s on
GPU. Queries are ~40 ms, so search has no `--gpu`.

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
