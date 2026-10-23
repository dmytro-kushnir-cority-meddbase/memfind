using System.Globalization;
using MemFind;

args = BackgroundProcess.Initialize(args);

var store = Store.DefaultRoot;
var k = 5;
var words = new List<string>();

var chatsOnly = false;
var memoryOnly = false;
var rebuild = false;
var includeAgents = true;
var sources = TranscriptSources.All;
var fp32 = false;
var gpu = false;
string? model = null;
var chats = false;
var json = false;
var context = 3;
var recency = new RecencyOptions();
var budgetMs = 7000;
var idleMinutes = 30;

for (var i = 0; i < args.Length; i++)
{
    switch (args[i])
    {
        case "--help" or "-h" or "/?":
            Usage();
            return 0;

        case "--store":
            if (i + 1 >= args.Length)
            {
                Console.Error.WriteLine("--store needs a path");
                return 2;
            }
            store = Paths.Expand(args[++i]);
            break;

        case "--claude-projects" or "--codex-sessions" or "--codex-memories":
            var flag = args[i];
            if (i + 1 >= args.Length)
            {
                Console.Error.WriteLine($"{flag} needs a path");
                return 2;
            }
            var path = Paths.Expand(args[++i]);
            if (flag == "--claude-projects")
            {
                Paths.ClaudeProjects = path;
            }
            else if (flag == "--codex-sessions")
            {
                Paths.CodexSessions = path;
            }
            else
            {
                Paths.CodexMemories = path;
            }
            break;

        case "-k":
            if (i + 1 >= args.Length || !int.TryParse(args[++i], out k) || k < 1)
            {
                Console.Error.WriteLine("-k needs a positive integer");
                return 2;
            }
            break;

        case "--chats-only":
            chatsOnly = true;
            break;

        case "--memory-only" or "--memories":
            memoryOnly = true;
            break;

        case "--rebuild":
            rebuild = true;
            break;

        case "--include-agents":
            break;

        case "--no-agents":
            includeAgents = false;
            break;

        case "--sources":
            sources = 0;
            foreach (var source in i + 1 < args.Length ? args[++i].Split(',') : Array.Empty<string>())
            {
                if (source.Trim().Equals("claude", StringComparison.OrdinalIgnoreCase))
                    sources |= TranscriptSources.Claude;
                else if (source.Trim().Equals("codex", StringComparison.OrdinalIgnoreCase))
                    sources |= TranscriptSources.Codex;
                else
                {
                    Console.Error.WriteLine($"--sources needs a comma-separated list of claude, codex (got '{source}')");
                    return 2;
                }
            }
            if (sources == 0)
            {
                Console.Error.WriteLine("--sources needs a comma-separated list of claude, codex");
                return 2;
            }
            break;

        case "--fp32":
            fp32 = true;
            break;

        case "--gpu":
            gpu = true;
            break;

        case "--offline":
            Embedder.AllowDownloads = false;
            break;

        case "--model":
            if (i + 1 >= args.Length)
            {
                Console.Error.WriteLine($"--model needs one of {ModelSpec.Keys}|both");
                return 2;
            }
            model = args[++i];
            break;

        case "--recency-boost":
            if (i + 1 >= args.Length
                || !double.TryParse(args[++i], NumberStyles.Float, CultureInfo.InvariantCulture, out var boost)
                || !double.IsFinite(boost) || boost < 0)
            {
                Console.Error.WriteLine("--recency-boost needs a non-negative number");
                return 2;
            }
            recency = recency with { Boost = boost };
            break;

        case "--recency-half-life":
            if (i + 1 >= args.Length
                || !double.TryParse(args[++i], NumberStyles.Float, CultureInfo.InvariantCulture, out var halfLife)
                || !double.IsFinite(halfLife) || halfLife <= 0)
            {
                Console.Error.WriteLine("--recency-half-life needs a positive number of days");
                return 2;
            }
            recency = recency with { HalfLifeDays = halfLife };
            break;

        case "--budget-ms":
            if (i + 1 >= args.Length || !int.TryParse(args[++i], out budgetMs) || budgetMs < 1)
            {
                Console.Error.WriteLine("--budget-ms needs a positive integer");
                return 2;
            }
            break;

        case "--idle-minutes":
            if (i + 1 >= args.Length || !int.TryParse(args[++i], out idleMinutes) || idleMinutes < 1)
            {
                Console.Error.WriteLine("--idle-minutes needs a positive integer");
                return 2;
            }
            break;

        case "--threads":
            if (i + 1 >= args.Length || !int.TryParse(args[++i], out var threads) || threads < 1)
            {
                Console.Error.WriteLine("--threads needs a positive integer");
                return 2;
            }
            Embedder.Threads = threads;
            break;

        case "-c":
            if (i + 1 >= args.Length || !int.TryParse(args[++i], out context) || context < 0)
            {
                Console.Error.WriteLine("-c needs a non-negative integer");
                return 2;
            }
            break;

        case "--json":
            json = true;
            break;

        case "--chats":
            chats = true;
            break;

        case "--all":
            break;

        default:
            words.Add(args[i]);
            break;
    }
}

if (fp32)
{
    if (model is not null && model != "bge-fp32")
    {
        Console.Error.WriteLine($"--fp32 and --model {model} disagree");
        return 2;
    }
    model = "bge-fp32";
}

var allowedKeys = Environment.GetEnvironmentVariable("MEMFIND_MODELS");
var enabled = string.IsNullOrWhiteSpace(allowedKeys) ? ModelSpec.Indexes
    : allowedKeys.Split(',', StringSplitOptions.TrimEntries | StringSplitOptions.RemoveEmptyEntries)
        .Select(ModelSpec.ByKey).Distinct().ToArray();
if (enabled.Length == 0 || enabled.Any(spec => spec is null))
{
    Console.Error.WriteLine($"MEMFIND_MODELS needs a comma-separated list of {ModelSpec.Keys}");
    return 2;
}
var enabledModels = enabled.Select(spec => spec!).ToArray();
var indexHook = words.Count == 1 && words[0] is "index-hook" or "index-worker";
if (indexHook && model is null)
    model = "gemma";

ModelSpec[] specs;
if (model is null)
{
    specs = [enabledModels.Contains(ModelSpec.Default) ? ModelSpec.Default : enabledModels[0]];
}
else if (model == "both")
{
    specs = enabledModels;
}
else if (ModelSpec.ByKey(model) is { } picked)
{
    if (!enabledModels.Contains(picked))
    {
        Console.Error.WriteLine($"Model {model} is disabled by MEMFIND_MODELS");
        return 2;
    }
    specs = [picked];
}
else
{
    Console.Error.WriteLine($"--model needs one of {ModelSpec.Keys}|both");
    return 2;
}

if (words.Count == 0)
{
    Usage();
    return 0;
}

using var stdout = new BufferedStream(Console.OpenStandardOutput());

if (words.Count == 1 && words[0] == "context-hook")
{
    // The hook's own default is three hits per source, not the CLI's five.
    return ContextHook.Run(store, model is null ? enabledModels : specs, recency,
        args.Contains("-k") ? k : 3, budgetMs, stdout);
}

if (words.Count == 1 && words[0] == "serve")
{
    if (!enabledModels.Contains(ModelSpec.Gemma))
    {
        Console.Error.WriteLine("Gemma is disabled by MEMFIND_MODELS");
        return 2;
    }
    return ContextHook.Serve(idleMinutes);
}

if (indexHook)
{
    // Capture effective options, including source roots, before starting an independent worker.
    var options = new List<string>
    {
        "--store", store, "--model", specs.Length == 1 ? specs[0].Key : "both",
        "--sources", sources == TranscriptSources.All ? "claude,codex" : sources == TranscriptSources.Claude ? "claude" : "codex",
        "--claude-projects", Paths.ClaudeProjects, "--codex-sessions", Paths.CodexSessions,
        "--codex-memories", Paths.CodexMemories, "--threads", Embedder.Threads.ToString(CultureInfo.InvariantCulture),
        "--offline",
    };
    if (chatsOnly) options.Add("--chats-only");
    if (memoryOnly) options.Add("--memory-only");
    if (rebuild) options.Add("--rebuild");
    if (!includeAgents) options.Add("--no-agents");
    if (gpu) options.Add("--gpu");
    Embedder.AllowDownloads = false;
    return IndexHook.Run(store, options.ToArray(), BuildIndexes, words[0] == "index-worker");
}

if (words.Count == 1 && words[0] == "index")
{
    Directory.CreateDirectory(store);
    using var indexLock = IndexLock.TryAcquire(store);
    if (indexLock is null)
    {
        Console.Error.WriteLine("memfind: another index is running, skipping");
        return 0;
    }

    return BuildIndexes();
}

if (words.Count == 1 && words[0] == "stats")
    return Stats(store, json, stdout, enabledModels);

int BuildIndexes()
{
    var code = 0;
    foreach (var spec in specs)
    {
        if (!chatsOnly)
            code = MemFind.Index.Build(store, spec, gpu, sources);
        if (code == 0 && !memoryOnly)
            code = ChatIndex.Build(store, rebuild, includeAgents, sources, spec, gpu);
        if (code != 0)
            break;
    }
    return code;
}

if (words.Count > 0 && words[0] == "show")
{
    if (words.Count != 2)
    {
        Console.Error.WriteLine("show needs one <file>:<line> anchor, as printed by a chat hit");
        return 2;
    }

    // A Windows path carries its own colon after the drive letter, so the anchor is the last one.
    var anchor = words[1];
    var colon = anchor.LastIndexOf(':');
    if (colon <= 0 || !long.TryParse(anchor[(colon + 1)..], out var showLine) || showLine < 1)
    {
        Console.Error.WriteLine($"not a <file>:<line> anchor: {anchor}");
        return 2;
    }

    return Show.Run(
        anchor[..colon],
        showLine,
        context,
        json,
        includeAgents || ChatIndex.LoadState(store, specs[0]).IncludeAgents,
        stdout
    );
}

chatsOnly |= chats;
if (chatsOnly && memoryOnly)
{
    Console.Error.WriteLine("--chats/--chats-only and --memory-only/--memories exclude each other");
    return 2;
}

var query = string.Join(' ', words);
var embedders = new Dictionary<ModelSpec, Task<Embedder>>();

try
{
    if (chatsOnly)
        return ChatIndex.Search(store, specs, query, k, json, embedders, stdout);
    if (memoryOnly)
        return MemFind.Index.Search(store, specs, query, k, json, embedders, stdout);

    var code = MemFind.Index.Search(store, specs, query, k, json, embedders, stdout);
    if (!json)
        Console.WriteLine();
    var chatCode = ChatIndex.Search(store, specs, query, k, json, embedders, stdout);
    return code == 0 || chatCode == 0 ? 0 : 1;
}
finally
{
    Embedder.DisposeAll(embedders);
}

static int Stats(string store, bool json, Stream stdout, ModelSpec[] enabledModels)
{
    var counts = enabledModels.Select(spec => Count(store, spec)).ToList();
    var main = counts[0];

    if (json)
    {
        using var writer = new System.Text.Json.Utf8JsonWriter(stdout);
        writer.WriteStartObject();
        writer.WriteString("store"u8, store);
        writer.WriteNumber("memories"u8, main.Memories);
        writer.WriteNumber("memoryBytes"u8, ChatIndex.Size(main.IndexPath));
        writer.WriteString("memoryModel"u8, main.MemoryModel);
        writer.WriteNumber("chatItems"u8, main.ChatItems);
        writer.WriteNumber("transcripts"u8, main.Transcripts);
        writer.WriteString("chatModel"u8, main.State.Model);
        writer.WriteBoolean("includeAgents"u8, main.State.IncludeAgents);
        writer.WriteNumber("metaVersion"u8, main.State.MetaVersion);
        writer.WriteNumber("vecBytes"u8, main.VecSize);
        writer.WriteNumber("metaBytes"u8, ChatIndex.Size(ChatIndex.MetaPath(store, main.Spec)));
        writer.WriteNumber("stateBytes"u8, ChatIndex.Size(ChatIndex.StatePath(store, main.Spec)));
        writer.WriteNumber("threads"u8, Embedder.Threads);
        writer.WriteNumber("batch"u8, Embedder.BatchSize);
        writer.WriteNumber("cores"u8, Environment.ProcessorCount);
        writer.WriteStartArray("models"u8);
        foreach (var c in counts)
        {
            if (c.Spec != main.Spec && c.Memories == 0 && c.ChatItems == 0)
                continue;
            writer.WriteStartObject();
            writer.WriteString("key"u8, c.Spec.Key);
            writer.WriteNumber("memories"u8, c.Memories);
            writer.WriteNumber("memoryBytes"u8, ChatIndex.Size(c.IndexPath));
            writer.WriteString("memoryModel"u8, c.MemoryModel);
            writer.WriteNumber("chatItems"u8, c.ChatItems);
            writer.WriteNumber("transcripts"u8, c.Transcripts);
            writer.WriteString("chatModel"u8, c.State.Model);
            writer.WriteNumber("vecBytes"u8, c.VecSize);
            writer.WriteEndObject();
        }
        writer.WriteEndArray();
        writer.WriteEndObject();
        writer.Flush();
        stdout.WriteByte((byte)'\n');
        return 0;
    }

    Console.WriteLine($"store          {store}");
    Console.WriteLine(
        $"memories       {main.Memories} items, {ChatIndex.Size(main.IndexPath) / 1024} KB, model {main.MemoryModel}"
    );
    Console.WriteLine(
        $"chats          {main.ChatItems} items from {main.Transcripts} transcripts, model {(main.State.Model.Length == 0 ? "(none)" : main.State.Model)}, meta v{main.State.MetaVersion}"
    );
    Console.WriteLine(
        $"               agents {(main.State.IncludeAgents ? "included" : "skipped")}"
    );
    Console.WriteLine($"chats.vec      {main.VecSize / 1048576} MB");
    Console.WriteLine(
        $"chats.meta     {ChatIndex.Size(ChatIndex.MetaPath(store, main.Spec)) / 1048576} MB"
    );
    Console.WriteLine(
        $"chats.state    {ChatIndex.Size(ChatIndex.StatePath(store, main.Spec)) / 1024} KB"
    );

    foreach (var c in counts)
    {
        if (c.Spec == main.Spec || (c.Memories == 0 && c.ChatItems == 0))
            continue;
        Console.WriteLine(
            $"{("memories." + c.Spec.Key).PadRight(14)} {c.Memories} items, {ChatIndex.Size(c.IndexPath) / 1024} KB, model {c.MemoryModel}"
        );
        Console.WriteLine(
            $"{("chats." + c.Spec.Key).PadRight(14)} {c.ChatItems} items from {c.Transcripts} transcripts, {c.VecSize / 1048576} MB vectors"
        );
    }

    Console.WriteLine(
        $"threads        {Embedder.Threads} intra-op, {Embedder.Threads} parse, 1 session, batch {Embedder.BatchSize} ({Environment.ProcessorCount} cores)"
    );
    return 0;
}

static Counts Count(string store, ModelSpec spec)
{
    var indexPath = MemFind.Index.PathFor(store, spec);
    var memories = 0;
    var memoryModel = "(none)";

    if (File.Exists(indexPath))
    {
        using var document = System.Text.Json.JsonDocument.Parse(File.ReadAllText(indexPath));
        if (document.RootElement.TryGetProperty("items", out var items))
            memories = items.GetArrayLength();
        if (document.RootElement.TryGetProperty("model", out var model))
            memoryModel = model.GetString() ?? "";
    }

    var state = ChatIndex.LoadState(store, spec);
    var vecSize = ChatIndex.Size(ChatIndex.VecPath(store, spec));
    var dim = state.Dim > 0 ? state.Dim : spec.Dim;
    var tracked = state
        .Files.Select(f => f.Path)
        .Distinct(StringComparer.OrdinalIgnoreCase)
        .Count();

    return new Counts(
        spec,
        indexPath,
        memories,
        memoryModel,
        state,
        vecSize,
        vecSize / (dim * sizeof(float)),
        tracked
    );
}

static void Usage()
{
    Console.WriteLine(
        """
        memfind — semantic search over memories, Claude Code and Codex transcripts

          memfind index [options]              build both indexes (<store>/.memfind/)
          memfind index-hook [options]         enqueue detached incremental indexing (Gemma by default);
                                               SessionStart/PreCompact hook event on stdin
          memfind <query words...> [-k N]      search memories, chats and agent chats; default k=5
          memfind ... --model bge|gemma|both   which model to index/search with
          memfind <query words...> --chats     search chat transcripts only (main and agent)
          memfind <query words...> --memories  search memories only
          memfind show <file>:<line> [-c N]    print that transcript message in full
          memfind stats                        what is indexed
          memfind context-hook [options]       Claude Code / Codex prompt hook (stdin/stdout JSON);
                                               --model defaults to both, -k to 3 per source
          memfind serve [--idle-minutes N]     keep gemma's query session loaded for context-hook;
                                               context-hook starts it when absent; exits after
                                               N idle minutes (default 30)
          memfind --help                       this screen

        Index options:
          --chats-only     skip the memory store
          --memory-only    skip the transcripts
          --rebuild        ignore the incremental state and re-read every transcript
          --no-agents      skip subagent transcripts (indexed by default)
          --sources <list> claude, codex or claude,codex (default: both); codex covers its
                           rollouts and memories. Codex can import Claude chats, so
                           `index --sources claude` avoids indexing them twice
          --fp32           alias for --model bge-fp32
          --gpu            try the CUDA execution provider, fall back to CPU
          --threads N      intra-op and parse threads (default: half the cores)

        Options:
          --store <path>   memory/index root (MEMFIND_STORE or local app data/memfind/memory)
          --claude-projects <path>  Claude transcripts (default: ~/.claude/projects)
          --codex-sessions <path>   Codex rollouts (default: $CODEX_HOME/sessions or ~/.codex/sessions)
          --codex-memories <path>   Codex memory (default: $CODEX_HOME/memories or ~/.codex/memories)
          -k N             number of hits to print per source (default: 5)
          -c N             messages of context on each side of a shown message (default: 3)
          --json           one JSON object per line instead of columns
          --offline        fail instead of downloading missing model files
          --budget-ms N    context-hook only: give up silently N ms after process start
                           (default: 7000)
          --recency-boost <score>   context-hook only: additive RRF bonus for chats
                           (default: 0.001; 0 disables; memories never decay)
          --recency-half-life <days>   context-hook only: age at half the power-law bonus
                           (default: 21, > 0); bonus = boost / sqrt(1 + 3 * age / days)
          --model <name>   bge (default), bge-fp32, gemma, or both
                           MEMFIND_MODELS limits enabled models; the first enabled model
                           is the default when bge is disabled; both uses enabled models
          --chats, --chats-only     search chat transcripts only (main and agent)
          --memories, --memory-only search memories only
          --all            search everything (the default)

        Every hit names its source: memory, chat/agent (Claude), or codex/codex-agent.
        --json carries the same value in "source". Paths support ~ and native separators.

        A chat hit prints three lines: the score row, the transcript path with the
        1-based line number of the message, then a 160-char preview. Feed that
        <path>:<line> straight back to `memfind show`.

        Exit codes: 0 ok, 1 missing/old index or an anchor that is not a message,
        2 bad arguments.

        Models: bge-small-en-v1.5 (384-dim, int8, the default) and embeddinggemma-300m
        (768-dim, int8), cached in local app data/memfind/, downloaded from
        huggingface.co on first use. Each keeps its own index files; gemma's carry a
        .gemma suffix. --model both searches each and fuses the two rankings (RRF),
        printing the fused score. Indexing reads both Claude and Codex session roots unless --sources narrows it.

        Examples:
          memfind index
          memfind how do I switch which clone IIS serves
          memfind "unsigned installer" -k 10
          memfind NUL-filled build output --chats
          memfind junction scope dir --memories
          memfind show ~/.codex/sessions/2026/10/22/rollout-example.jsonl:42 -c 2
        """
    );
}

internal sealed record Counts(
    ModelSpec Spec,
    string IndexPath,
    int Memories,
    string MemoryModel,
    ChatState State,
    long VecSize,
    long ChatItems,
    int Transcripts
);
