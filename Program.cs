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

for (var i = 0; i < args.Length; i++)
{
    switch (args[i])
    {
        case "--help" or "-h" or "/?":
            Usage();
            return 0;

        case "--store":
            if (i + 1 >= args.Length) { Console.Error.WriteLine("--store needs a path"); return 2; }
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
        if (!chatsOnly) code = MemFind.Index.Build(store, !fp32);
        if (code == 0 && !memoryOnly) code = ChatIndex.Build(store, rebuild, includeAgents, !fp32);
        return code;
    }
    finally
    {
        mutex.ReleaseMutex();
    }
}

if (words.Count == 1 && words[0] == "stats") return Stats(store);

var query = string.Join(' ', words);

if (all)
{
    var code = MemFind.Index.Search(store, query, k);
    Console.WriteLine();
    return ChatIndex.Search(store, query, k) is var chatCode && code != 0 ? code : chatCode;
}

return chats ? ChatIndex.Search(store, query, k) : MemFind.Index.Search(store, query, k);

static int Stats(string store)
{
    var indexPath = MemFind.Index.PathFor(store);
    var memories = 0;
    var memoryModel = "(none)";

    if (File.Exists(indexPath))
    {
        using var document = System.Text.Json.JsonDocument.Parse(File.ReadAllText(indexPath));
        if (document.RootElement.TryGetProperty("items", out var items)) memories = items.GetArrayLength();
        if (document.RootElement.TryGetProperty("model", out var model)) memoryModel = model.GetString() ?? "";
    }

    var state = ChatIndex.LoadState(store);
    var vecSize = ChatIndex.Size(ChatIndex.VecPath(store));
    var chatItems = vecSize / (Embedder.Dim * sizeof(float));
    var tracked = state.Files.Select(f => f.Path).Distinct(StringComparer.OrdinalIgnoreCase).Count();

    Console.WriteLine($"store          {store}");
    Console.WriteLine($"memories       {memories} items, {ChatIndex.Size(indexPath) / 1024} KB, model {memoryModel}");
    Console.WriteLine($"chats          {chatItems} items from {tracked} transcripts, model {(state.Model.Length == 0 ? "(none)" : state.Model)}");
    Console.WriteLine($"               agents {(state.IncludeAgents ? "included" : "skipped")}");
    Console.WriteLine($"chats.vec      {vecSize / 1048576} MB");
    Console.WriteLine($"chats.meta     {ChatIndex.Size(ChatIndex.MetaPath(store)) / 1048576} MB");
    Console.WriteLine($"chats.state    {ChatIndex.Size(ChatIndex.StatePath(store)) / 1024} KB");
    Console.WriteLine($"threads        {Embedder.Threads} intra-op, {Embedder.Threads} parse, 1 session, batch {Embedder.BatchSize} ({Environment.ProcessorCount} cores)");
    return 0;
}

static void Usage()
{
    Console.WriteLine("""
        memfind — semantic search over the Claude Code memory store and chat transcripts

          memfind index [options]              build both indexes (<store>\.memfind\)
          memfind <query words...> [-k N]      search memories; default k=5
          memfind <query words...> --chats     search chat transcripts
          memfind <query words...> --all       search both
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

        Model: bge-small-en-v1.5 (384-dim), int8 by default, cached in
        %LOCALAPPDATA%\memfind\, downloaded from huggingface.co on first use.
        Transcripts are read from %USERPROFILE%\.claude\projects.

        Examples:
          memfind index
          memfind how do I switch which clone IIS serves
          memfind "unsigned installer" -k 10
          memfind NUL-filled build output --chats
        """);
}
