using MemFind;

var store = Store.DefaultRoot;
var k = 5;
var words = new List<string>();

var chatsOnly = false;
var memoryOnly = false;
var rebuild = false;
var includeAgents = false;
var fp32 = false;
var chats = false;
var all = false;
var json = false;
var context = 3;

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
            store = args[++i];
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

        case "--memory-only":
            memoryOnly = true;
            break;

        case "--rebuild":
            rebuild = true;
            break;

        case "--include-agents":
            includeAgents = true;
            break;

        case "--fp32":
            fp32 = true;
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
            all = true;
            break;

        default:
            words.Add(args[i]);
            break;
    }
}

if (words.Count == 0)
{
    Usage();
    return 0;
}

if (words.Count == 1 && words[0] == "index")
{
    using var mutex = new Mutex(false, @"Global\memfind-index");
    bool acquired;
    try
    {
        acquired = mutex.WaitOne(0);
    }
    catch (AbandonedMutexException)
    {
        acquired = true;
    }

    if (!acquired)
    {
        Console.Error.WriteLine("memfind: another index is running, skipping");
        return 0;
    }

    try
    {
        var code = 0;
        if (!chatsOnly)
            code = MemFind.Index.Build(store, !fp32);
        if (code == 0 && !memoryOnly)
            code = ChatIndex.Build(store, rebuild, includeAgents, !fp32);
        return code;
    }
    finally
    {
        mutex.ReleaseMutex();
    }
}

if (words.Count == 1 && words[0] == "stats")
    return Stats(store, json);

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
        includeAgents || ChatIndex.LoadState(store).IncludeAgents
    );
}

var query = string.Join(' ', words);

if (all)
{
    var code = MemFind.Index.Search(store, query, k, json);
    if (!json)
        Console.WriteLine();
    return ChatIndex.Search(store, query, k, json) is var chatCode && code != 0 ? code : chatCode;
}

return chats
    ? ChatIndex.Search(store, query, k, json)
    : MemFind.Index.Search(store, query, k, json);

static int Stats(string store, bool json)
{
    var indexPath = MemFind.Index.PathFor(store);
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

    var state = ChatIndex.LoadState(store);
    var vecSize = ChatIndex.Size(ChatIndex.VecPath(store));
    var chatItems = vecSize / (Embedder.Dim * sizeof(float));
    var tracked = state
        .Files.Select(f => f.Path)
        .Distinct(StringComparer.OrdinalIgnoreCase)
        .Count();

    if (json)
    {
        using var output = new JsonLines();
        var writer = output.Writer;
        writer.WriteStartObject();
        writer.WriteString("store"u8, store);
        writer.WriteNumber("memories"u8, memories);
        writer.WriteNumber("memoryBytes"u8, ChatIndex.Size(indexPath));
        writer.WriteString("memoryModel"u8, memoryModel);
        writer.WriteNumber("chatItems"u8, chatItems);
        writer.WriteNumber("transcripts"u8, tracked);
        writer.WriteString("chatModel"u8, state.Model);
        writer.WriteBoolean("includeAgents"u8, state.IncludeAgents);
        writer.WriteNumber("metaVersion"u8, state.MetaVersion);
        writer.WriteNumber("vecBytes"u8, vecSize);
        writer.WriteNumber("metaBytes"u8, ChatIndex.Size(ChatIndex.MetaPath(store)));
        writer.WriteNumber("stateBytes"u8, ChatIndex.Size(ChatIndex.StatePath(store)));
        writer.WriteNumber("threads"u8, Embedder.Threads);
        writer.WriteNumber("batch"u8, Embedder.BatchSize);
        writer.WriteNumber("cores"u8, Environment.ProcessorCount);
        writer.WriteEndObject();
        output.Flush();
        return 0;
    }

    Console.WriteLine($"store          {store}");
    Console.WriteLine(
        $"memories       {memories} items, {ChatIndex.Size(indexPath) / 1024} KB, model {memoryModel}"
    );
    Console.WriteLine(
        $"chats          {chatItems} items from {tracked} transcripts, model {(state.Model.Length == 0 ? "(none)" : state.Model)}, meta v{state.MetaVersion}"
    );
    Console.WriteLine($"               agents {(state.IncludeAgents ? "included" : "skipped")}");
    Console.WriteLine($"chats.vec      {vecSize / 1048576} MB");
    Console.WriteLine($"chats.meta     {ChatIndex.Size(ChatIndex.MetaPath(store)) / 1048576} MB");
    Console.WriteLine($"chats.state    {ChatIndex.Size(ChatIndex.StatePath(store)) / 1024} KB");
    Console.WriteLine(
        $"threads        {Embedder.Threads} intra-op, {Embedder.Threads} parse, 1 session, batch {Embedder.BatchSize} ({Environment.ProcessorCount} cores)"
    );
    return 0;
}

static void Usage()
{
    Console.WriteLine(
        """
        memfind — semantic search over the Claude Code memory store and chat transcripts

          memfind index [options]              build both indexes (<store>\.memfind\)
          memfind <query words...> [-k N]      search memories; default k=5
          memfind <query words...> --chats     search chat transcripts
          memfind <query words...> --all       search both
          memfind show <file>:<line> [-c N]    print that transcript message in full
          memfind stats                        what is indexed
          memfind --help                       this screen

        Index options:
          --chats-only     skip the memory store
          --memory-only    skip the transcripts
          --rebuild        ignore the incremental state and re-read every transcript
          --include-agents index subagent transcripts too
          --fp32           use the fp32 model instead of int8
          --threads N      intra-op and parse threads (default: half the cores)

        Options:
          --store <path>   memory store root (default: C:\Git\docs\memory)
          -k N             number of hits to print per source (default: 5)
          -c N             messages of context on each side of a shown message (default: 3)
          --json           one JSON object per line instead of columns

        A chat hit prints three lines: the score row, the transcript path with the
        1-based line number of the message, then a 160-char preview. Feed that
        <path>:<line> straight back to `memfind show`.

        Exit codes: 0 ok, 1 missing/old index or an anchor that is not a message,
        2 bad arguments.

        Model: bge-small-en-v1.5 (384-dim), int8 by default, cached in
        %LOCALAPPDATA%\memfind\, downloaded from huggingface.co on first use.
        Transcripts are read from %USERPROFILE%\.claude\projects.

        Examples:
          memfind index
          memfind how do I switch which clone IIS serves
          memfind "unsigned installer" -k 10
          memfind NUL-filled build output --chats
          memfind show C:\Users\me\.claude\projects\C--Git\a1b2.jsonl:8412 -c 2
        """
    );
}
