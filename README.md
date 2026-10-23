# memfind

Semantic search over a Markdown memory store, Codex memory, and local Claude Code and Codex transcripts.
One NativeAOT executable, no service, no server. Embeddings run locally; model weights are downloaded on first use.

## Run locally

Requires the .NET 10 SDK. On macOS the CPU ONNX Runtime package is selected automatically;
Windows and Linux retain the GPU package (CPU inference also works with it).
Intel macOS selects ONNX Runtime 1.23.2 because the newer packages no longer contain
`osx-x64` native binaries; Apple Silicon keeps 1.30.0.

```sh
dotnet build -c Release
dotnet run -c Release --no-build -- index
dotnet run -c Release --no-build -- "what did we decide about deployment"
dotnet run -c Release --no-build -- stats --json
python3 tests/smoke.py
# Or test a published executable: python3 tests/smoke.py path/to/publish/memfind
# Two-model regression (downloads Gemma too): python3 tests/models.py path/to/publish/memfind
```

The smoke check creates isolated Claude/Codex/memory fixtures and runs real bge embeddings,
search, show, incremental updates, agent filtering, deletions and memory edits. It needs the
first-use model download, and does not index or change your real histories.

To publish a native executable for the current platform: `dotnet publish -c Release`.

## Source paths

Paths use the current OS conventions and support `~` in CLI arguments. A missing source is
skipped when another transcript root exists. The default store is created on `index`.

| Source | Default | Override |
| --- | --- | --- |
| Memory/index store | .NET local application data directory + `memfind/memory` | `--store PATH` or `MEMFIND_STORE` |
| Claude chats | `~/.claude/projects` | `--claude-projects PATH` |
| Codex chats | `~/.codex/sessions` | `--codex-sessions PATH` |
| Codex memory | `~/.codex/memories` | `--codex-memories PATH` |

`CODEX_HOME` replaces `~/.codex` for both Codex roots. Existing Windows memory stores can
still be selected with `--store C:\Git\docs\memory`.

Codex memory includes `MEMORY.md`, `memory_summary.md`, `rollout_summaries/**/*.md` and
`extensions/ad_hoc/notes/**/*.md`. It is chunked into passages with original file and line
anchors, in the same memory index as the ordinary Markdown store. Raw memory intermediates
and skill files are excluded.

Codex sessions use `session_meta` for session/cwd/agent identity and canonical
`response_item` user/assistant messages for text. Tool results, reasoning, system/developer
messages and duplicate `event_msg` records are excluded. Short Codex messages are retained.
`--no-agents` also excludes Codex rollouts whose session metadata identifies a subagent.
Both `.jsonl` and compressed `.jsonl.zst` sessions are read directly, including by `show`.
Archive anchors use decompressed line numbers. Unchanged archives are skipped; a changed
archive is replaced in the index in full. No external `zstd` command is needed.
Source transcripts and memories are read only; indexes are written only under the store.

## Commands

| Command | What it does |
| --- | --- |
| `memfind index [options]` | Build both indexes into `<store>\.memfind\` |
| `memfind <query words...>` | Search memories, chats and agent chats (memories first) |
| `memfind <query words...> --chats` | Search chat transcripts only (main and agent) |
| `memfind <query words...> --memories` | Search memories only |
| `memfind show <file>:<line> [-c N]` | Print that transcript message in full, with neighbours |
| `memfind stats` | What is indexed |
| `memfind context-hook` | Claude Code / Codex UserPromptSubmit hook, JSON on stdin/stdout |
| `memfind serve` | Keep gemma's query session loaded; `context-hook` asks it over a per-user named pipe, starts one when none is listening, and embeds in-process until it answers |
| `memfind --help` | Usage |

Every command takes `--model bge|bge-fp32|gemma|both`; `bge` is the default. On `index`, `both`
builds each model in turn; on a query it searches with each and fuses the two rankings.

## Flags

| Flag | Applies to | Meaning |
| --- | --- | --- |
| `--store <path>` | all | Memory/index store root (see Source paths) |
| `--claude-projects <path>` | all | Claude transcript root |
| `--codex-sessions <path>` | all | Codex transcript root |
| `--codex-memories <path>` | all | Codex memory root |
| `-k N` | queries | Hits per source (default 5) |
| `-c N` | `show` | Kept messages of context on each side (default 3) |
| `--json` | queries, `show`, `stats` | JSONL instead of columns |
| `--chats` / `--chats-only` | queries | Search chat transcripts only (main and agent) |
| `--memories` / `--memory-only` | queries | Search memories only; with `--chats` it is an error (exit 2) |
| `--all` | queries | Search everything — the default, accepted for old callers |
| `--chats-only` / `--memory-only` | `index` | Skip the other half |
| `--rebuild` | `index` | Ignore the incremental state, re-read every transcript |
| `--no-agents` | `index` | Skip subagent transcripts and sidechain messages (indexed by default; `--include-agents` is accepted and does nothing) |
| `--sources <list>` | `index` | `claude`, `codex` or `claude,codex` (default both). Codex means its rollouts and its memories. Changing it rebuilds the chat index and rewrites the memory index; on other commands it is validated and ignored. E.g. `memfind index --sources claude`: Codex can import Claude chats, which indexes them twice |
| `--model <name>` | `index`, queries, `stats` | `bge` (default), `bge-fp32`, `gemma`, or `both` |
| `--fp32` | `index` | Alias for `--model bge-fp32`; with `--model both` it is an error (exit 2) |
| `--gpu` | `index` | Try the CUDA execution provider, fall back to CPU |
| `--threads N` | all | Intra-op and parse threads (default: half the cores; a search uses 4 intra-op) |
| `--offline` | all | Fail instead of downloading missing model files |
| `--budget-ms N` | `context-hook` | Give up silently N ms after process start (default 7000) |
| `--idle-minutes N` | `serve` | Exit after N minutes without a query (default 30) |

## Exit codes

| Code | When |
| --- | --- |
| 0 | Success |
| 1 | Missing or outdated index, missing transcript, or an anchor line that is not a kept message |
| 2 | Bad arguments |

## Output — text

The first line of every hit starts with the score, then the source padded to six characters:
`memory` (memory store), `chat`/`agent` (Claude main/subagent transcripts) or `codex`/`codex-agent` (Codex main/subagent rollouts). A default search prints the
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

One object per line (JSONL). Chat `ts` is the original transcript message timestamp;
memory `ts` is the source file's last-write time in UTC (absent in older indexes).
Text mode renders chat time locally. `score` is rounded to 4 decimals.

| Shape | Fields |
| --- | --- |
| Memory hit | `source="memory"`, `score`, `file`, `theme`, `title`, `description`, `ts`, `line` (Codex memory passage anchor; 0 for ordinary store documents) |
| Chat hit | `source="chat"`/`"agent"` (Claude) or `"codex"`/`"codex-agent"` (Codex), `score`, `file` (full transcript path), `line`, `uuid`, `chunk`, `session`, `cwd`, `role`, `ts`, `text` (the stored preview) |
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

All under `<store>/.memfind/` (with a `.gitignore` of `*`).

| File | Schema |
| --- | --- |
| `index.json` | `{model, dim, builtAt, items:[{file, theme, title, description, vec[dim]}]}` — the memory index, read whole |
| `chats.vec` | Raw little-endian `float32[dim]` per item, back to back; item *i* is at `i * dim * 4` bytes. Memory-mapped at search time |
| `chats.meta.jsonl` | One object per item, in `chats.vec` order: `{i, file, source, session, cwd, role, ts, line, uuid, chunk, text}`. `file` is an absolute native path; `text` is the 300-char preview |
| `chats.state.json` | `{model, dim, metaVersion, includeAgents, sources (1 claude, 2 codex, 3 both; absent means both), files:[{path, length, lines, lastWriteUtc, itemsFrom, itemsCount}]}` — the incremental state: bytes and lines already consumed per transcript (compressed file size for archives), and the item range they produced |

Index progress reports processed fragments and characters per second (current interval and
overall embedding average). Characters count the text supplied to the embedder, excluding
JSON records and tool output; the count uses .NET UTF-16 string length.

Each model keeps its own set. bge (and `bge-fp32`, which shares them) uses the names above;
gemma inserts its key: `index.gemma.json`, `chats.gemma.vec`, `chats.gemma.meta.jsonl`,
`chats.gemma.state.json`. `dim` is 384 for bge, 768 for gemma.

`metaVersion` is the layout of `chats.meta.jsonl` (currently **3**). Version 3 adds absolute paths and source tags for mixed Claude/Codex roots. A search against an older one
fails with `chats.meta.jsonl is old — run: memfind index --rebuild` and exit 1; `index` rebuilds
from scratch on the mismatch instead.

## Models

| Key | Model | Dim | Files in the OS local application data directory under `memfind/<model>/` |
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

`index --gpu` tries CUDA auto-detection on Windows and reports CPU fallback elsewhere. The package is `Microsoft.ML.OnnxRuntime.Gpu`,
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

## Prompt context hook (Claude Code and Codex)

`memfind context-hook` is the `UserPromptSubmit` hook for both Claude Code and
Codex. It reads the hook event on stdin, searches the existing memory and chat
indexes in-process and writes one JSON object,
`{"hookSpecificOutput":{"hookEventName":"UserPromptSubmit","additionalContext":...}}`.
It uses the `--model` value given on its command line (`bge`, `bge-fp32`, `gemma`
or `both`) and defaults to `both`, which fuses the bge and Gemma rankings (RRF) as
`memfind --model both` does.

`additionalContext` is one header line, then one line per hit: memories first,
then chat transcripts, at most `-k` (default 3) of each after the filters below.

```text
  0.81  memory  some_memory.md  Title - description
  0.90  memory  /home/me/.codex/memories/MEMORY.md:12  Heading - passage, whitespace collapsed, 160 chars...
  0.79  chat    /home/me/.claude/projects/p/session.jsonl:628  2026-10-22 assistant [/project] - text, 160 chars...
```

The number is the gate similarity (a model cosine, never the RRF score). Store
memories print their description whole; Codex memory passages and chat, agent,
codex and codex-agent messages are collapsed to one line and cut at 160 characters.
The hook skips prompts of fewer than three words, truncates the query to 2000
characters, and drops hits below the similarity floor, repeated texts and the
current session/transcript. The whole context is capped at 3000 characters; a
line that would exceed it is dropped. With nothing left it writes nothing.
Errors leave the prompt unchanged: no output, exit 0.

Both hosts allow a **ten-second timeout**. `--budget-ms N` (default 7000) is
counted from process start, so runtime startup is charged. When it runs out the
hook writes nothing and ends the process at once with exit code 0, without
waiting for a model load still inside ONNX Runtime. It never downloads or
indexes, and skips retrieval if any selected model file is missing/empty or a
selected model has neither a memory nor a chat index. Only already-built indexes
are searched; build the chat index separately to include transcripts.

The similarity floor is per model, on that model's cosine: bge and bge-fp32 0.70,
gemma 0.38. These are the values of the earlier single-model hooks, an initial
local probe setting, not calibrated confidence probabilities. In `both` mode a hit
passes when at least one of its legs clears its own model's floor, and its similarity
is the highest cosine among the legs that do. Memories keep their search-score order
(RRF when fused, cosine for a single model). The hook retrieves 64 candidates per
source, so this orders that candidate subset, not the whole corpus. Ordinary CLI
search keeps its existing ordering.

Chats and subagent chats get a small additive freshness bonus by default:
`rank = RRF + boost / sqrt(1 + 3 * age_days / half_life)`.
`--recency-boost <score>` is in RRF score units (default 0.001; 0 disables it).
`--recency-half-life <days>` is the age at which the bonus has halved (default 21,
must be greater than 0). With these defaults the bonus is 0.001 when fresh,
about 0.000707 after 7 days, and 0.0005 after 21 days. It has a gradual power-law
tail. Single-model chat hooks use `1 / (60 + retrieval_rank)` as their RRF base,
with ranks assigned before gating. The relevance gate always runs before the bonus.
Future dates are clamped to age zero; missing or invalid dates get no bonus.
To disable it: `memfind context-hook --recency-boost 0`.

Chat fragments use the original message timestamp; tool calls/results are excluded.
All memory sources, including Codex memory passages, stay free of decay regardless
of their file modification time. These options affect the context hook only;
ordinary CLI search keeps its existing ordering.

Publish and install the native executable, then merge this entry into your
Codex home's `hooks.json` (`$CODEX_HOME` or `~/.codex`), or for Claude Code into the
`hooks` of `~/.claude/settings.json` (same shape). Use an absolute executable
path; [hooks/hooks.example.json](hooks/hooks.example.json) is the same example:

```json
{
  "hooks": {
    "UserPromptSubmit": [{
      "hooks": [{
        "type": "command",
        "command": "/usr/local/bin/memfind context-hook --store /path/to/memory --model both",
        "timeout": 10,
        "additionalContextLimit": 3000
      }]
    }]
  }
}
```

On Windows use `commandWindows` with the native executable path quoted as needed.
Review and trust the hook in Codex `/hooks` or the app's Hooks settings, then
start a new chat. Codex skips untrusted hook definitions. Local command hooks
require local orchestration; see the [official hooks documentation](https://learn.chatgpt.com/docs/hooks).
Prepare indexes separately with `memfind index --model both` (or start with
`memfind index --model both --memory-only` for curated memory).

Hook checks: `dotnet run -c Release --project tests/HookContracts/HookContracts.csproj`
(native contract checks and a real worker that runs out its budget; no Python).

## Claude SessionStart hook

`~/.claude/settings.json` runs the index on every session start, async so it never blocks:

```json
{ "type": "command", "command": "C:\\Users\\dkushnir\\bin\\memfind\\memfind.exe",
  "args": ["index"], "async": true, "timeout": 300 }
```

## Index writer lock

`index` exclusively opens `.memfind/index.lock` without waiting. A second
indexer for the same store prints `memfind: another index is running, skipping`
and exits 0. Filesystem aliases share the same lock, while independent stores
can index concurrently, including a staging store for a long first build.
The lock file stays in place after release. Older binaries used a global named
mutex; stop any old indexer before indexing the same store with this version.
