using System.Diagnostics;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using MemFind;

if (args.Length == 2 && args[0] == "--input")
{
    using var stdout = Console.OpenStandardOutput();
    Environment.ExitCode = ContextHook.Run(Path.GetTempPath(), [ModelSpec.Gemma], new RecencyOptions(), 3,
        int.Parse(args[1]), stdout);
    return;
}
if (args.Length == 2 && args[0] == "--lock")
{
    try
    {
        using var held = IndexLock.TryAcquire(args[1]);
        Console.WriteLine(held is null ? "busy" : "acquired");
    }
    catch (IOException e)
    {
        Console.Error.WriteLine($"Lock I/O error: 0x{e.HResult:X}");
        throw;
    }
    return;
}

var root = Path.Combine(Path.GetTempPath(), "memfind-hook-contracts-" + Guid.NewGuid());
Directory.CreateDirectory(root);
try
{
    ModelSpec[] gemma = [ModelSpec.Gemma];
    ModelSpec[] both = ModelSpec.Indexes;
    var off = new RecencyOptions { Boost = 0 };
    var decay = new RecencyOptions();
    var current = Path.Combine(root, "current.jsonl");
    var evt = Element(new { session_id = "current", transcript_path = current });
    var rows = new[]
    {
        Chat("a", "below", .379f),
        Chat("b", "boundary", .381f),
        Chat("c", "highest", .9f),
        Chat("d", "current session", .99f, session: "current"),
        Chat(current, "current transcript", .99f),
        Chat("e", " HIGHEST  ", .8f),
    };
    var context = Build(chats: rows)!;
    var lines = context.Split('\n');
    Check(lines[0] == ContextHook.Header && ContextHook.Header.StartsWith("memfind: memories, past chats"), "header line");
    Check(lines.Skip(1).All(line => line.StartsWith("  ")), "one hit per indented line");
    Check(!context.Contains("below") && context.Contains("boundary"), "Gemma score floor");
    Check(context.IndexOf("highest", StringComparison.Ordinal) < context.IndexOf("boundary", StringComparison.Ordinal), "score order");
    Check(!context.Contains("current session") && !context.Contains("current transcript"), "exclude current source");
    Check(!context.Contains("HIGHEST"), "deduplication");
    Check(lines[1] == "  0.90  codex   c:17  2026-10-22 user [/project] - highest", "chat line format");
    Check(Build(chats: [Chat("a", "low", .1f)]) is null, "no irrelevant context");
    var spaced = Build(chats: [Chat("a", "line one\n\n  line\ttwo " + new string('x', 300), .9f)])!.Split('\n')[1];
    Check(spaced.EndsWith(" - line one line two " + new string('x', 160 - 18) + "..."), "chat text collapsed and cut at 160");
    var emoji = Build(chats: [Chat("a", new string('a', 159) + "\U0001F600 tail", .9f)])!.Split('\n')[1];
    Check(emoji.EndsWith(" - " + new string('a', 159) + "..."), "cut never splits a surrogate pair");

    context = Build(memories: [Memory("fact.md", "A fact — from the store.", .9f), Memory("fact2.md", "", .8f)])!;
    lines = context.Split('\n');
    Check(lines[1] == "  0.90  memory  fact.md  Title fact.md - A fact — from the store.", "memory line format");
    Check(lines[2] == "  0.80  memory  fact2.md  Title fact2.md - ", "memory without description is kept");
    var passage = Path.Combine(root, "codex", "MEMORY.md");
    context = Build(memories: [Memory(passage, "# Heading\n" + new string('y', 400), .9f, line: 12)])!;
    Check(context.Split('\n')[1] == $"  0.90  memory  {passage}:12  Title {passage} - # Heading {new string('y', 150)}...", "Codex memory passage line");

    var many = Enumerable.Range(0, 6).Select(i => Memory($"m{i}.md", "memory " + i, .9f - i / 100f)).ToArray();
    var chatsMany = Enumerable.Range(0, 6).Select(i => Chat($"c{i}", "chat " + i, .9f - i / 100f)).ToArray();
    lines = Build(memories: many, chats: chatsMany)!.Split('\n');
    Check(lines.Length == 7 && lines.Skip(1).Take(3).All(line => line.Contains("  memory  "))
        && lines.Skip(4).All(line => line.Contains("  codex   ")), "k hits per source, memories first");
    Check(lines[1].Contains("m0.md") && lines[3].Contains("m2.md") && lines[4].Contains("c0:"), "per-source rank order");
    Check(Build(memories: many, k: 1)!.Split('\n').Length == 2, "k is a parameter");
    Check(Build(memories: [Memory("a.md", "same body", .9f)], chats: [Chat("a", "Same  body", .9f)])!.Split('\n').Length == 2, "dedup spans sources");

    var now = DateTimeOffset.Parse("2026-10-22T12:00:00Z");
    var fresh = now.ToString("O");
    var old = now.AddDays(-365).ToString("O");
    context = Build(chats:
    [
        Chat("old", "older near match", .0300f, ts: old, legs: Legs(("gemma", .65f, 2))),
        Chat("fresh", "fresh near match", .0296f, ts: fresh, legs: Legs(("gemma", .63f, 3))),
        Chat("strong", "strong older match", .0320f, ts: old, legs: Legs(("gemma", .8f, 1))),
        Chat("irrelevant", "fresh irrelevant", .05f, ts: fresh, legs: Legs(("gemma", .379f, 1))),
    ], recency: decay, now: now)!;
    Check(context.IndexOf("strong older match", StringComparison.Ordinal) < context.IndexOf("fresh near match", StringComparison.Ordinal)
        && context.IndexOf("fresh near match", StringComparison.Ordinal) < context.IndexOf("older near match", StringComparison.Ordinal), "bounded freshness reranking");
    Check(!context.Contains("fresh irrelevant"), "freshness never bypasses similarity floor");
    Check(Build(chats: [Chat("a", "raw", .6f, ts: fresh)], recency: decay, now: now)!.Contains("  0.60  "), "printed similarity is the raw cosine");
    Check(decay.Boost == .001 && decay.HalfLifeDays == 21, "default chat decay settings");
    Check(Close(ContextHook.Rank(.03, now, decay, now), .031), "fresh additive RRF bonus");
    Check(ContextHook.Rank(.03, null, decay, now) == .03, "missing date has no bonus");
    Check(Close(ContextHook.Rank(.03, now.AddDays(3), decay, now), .031), "future date bonus is clamped");
    Check(Close(ContextHook.Rank(.03, now.AddDays(-7), decay, now), .03 + .001 / Math.Sqrt(2)), "seven day power law");
    Check(Close(ContextHook.Rank(.03, now.AddDays(-21), decay, now), .0305), "twenty-one day half bonus");
    foreach (var age in new[] { 90, 365 })
        Check(Close(ContextHook.Rank(.03, now.AddDays(-age), decay, now), .03 + .001 / Math.Sqrt(1 + age / 7d)), "power-law tail at " + age);
    Check(ContextHook.Rank(.03, now, off, now) == .03, "zero boost disables recency");
    Check(Close(ContextHook.Rank(.03, now, new RecencyOptions { Boost = .002 }, now), .032), "boost is a parameter");
    Check(Close(ContextHook.Rank(.03, now.AddDays(-1), new RecencyOptions { Boost = .002, HalfLifeDays = 1 }, now), .031), "half life is a parameter");
    foreach (var date in new[] { "invalid", "" })
    {
        context = Build(chats:
        [
            Chat("bad", "undated lower match", .0300f, ts: date, legs: Legs(("gemma", .6f, 2))),
            Chat("valid", "dated higher match", .0301f, ts: old, legs: Legs(("gemma", .65f, 1))),
        ], recency: decay, now: now)!;
        Check(context.IndexOf("dated higher match", StringComparison.Ordinal) < context.IndexOf("undated lower match", StringComparison.Ordinal), "invalid or empty date cannot add a bonus");
    }
    Check(Build(chats: [Chat("a", "bad date", .6f, ts: "invalid")], recency: decay, now: now)!.Contains(" invalid user "), "invalid date prints raw");

    foreach (var source in new[] { "chat", "agent", "codex", "codex-agent" })
    {
        var older = Chat("old", "older chat", .65f, ts: old, source: source);
        var newer = Chat("fresh", "newer chat", .63f, ts: fresh, source: source);
        context = Build(chats: [newer, older], recency: decay, now: now)!;
        Check(context.IndexOf("newer chat", StringComparison.Ordinal) < context.IndexOf("older chat", StringComparison.Ordinal), source + " uses one-leg RRF units");
        context = Build(chats: [newer, older], recency: off, now: now)!;
        Check(context.IndexOf("older chat", StringComparison.Ordinal) < context.IndexOf("newer chat", StringComparison.Ordinal), source + " with boost zero preserves cosine order");
    }

    var aging = Chat("aging", "aging chat", .03f, ts: old, legs: Legs(("gemma", .6f, 2)));
    aging.Row = aging.Row with { Time = fresh };
    var competitor = Chat("competitor", "stronger chat", .0305f, ts: old, legs: Legs(("gemma", .65f, 1)));
    context = Build(chats: [aging, competitor], recency: decay, now: now)!;
    Check(context.IndexOf("stronger chat", StringComparison.Ordinal) < context.IndexOf("aging chat", StringComparison.Ordinal), "original message age overrides other row dates");
    aging.Row = aging.Row with { Ts = fresh };
    context = Build(chats: [aging, competitor], recency: decay, now: now)!;
    Check(context.IndexOf("aging chat", StringComparison.Ordinal) < context.IndexOf("stronger chat", StringComparison.Ordinal), "same text with a recent message date earns freshness");
    context = Build(chats: [aging, competitor], recency: decay, now: now.AddDays(365))!;
    Check(context.IndexOf("stronger chat", StringComparison.Ordinal) < context.IndexOf("aging chat", StringComparison.Ordinal), "same message loses its near-tie advantage as the clock advances");

    // Excluded sources retain their retrieval ranks; a gate must not compress evidence.
    var excluded = Enumerable.Range(0, 30).Select(i => Chat("skip" + i, "skip " + i, .9f - i / 100f, session: "current")).ToList();
    excluded.Add(Chat("old", "old ranked match", .55f, ts: old));
    excluded.Add(Chat("fresh", "fresh ranked match", .53f, ts: fresh));
    context = Build(chats: excluded.ToArray(), recency: decay with { Boost = .0002 }, now: now)!;
    Check(context.IndexOf("fresh ranked match", StringComparison.Ordinal) < context.IndexOf("old ranked match", StringComparison.Ordinal), "single-model ranks precede exclusions");

    var memoryOld = Memory("old.md", "older memory", .65f, timestamp: now.AddYears(-20));
    var memoryFresh = Memory(passage, "fresh memory passage", .63f, line: 12, timestamp: now.AddDays(1));
    foreach (var settings in new[] { decay, decay with { Boost = 10 }, off })
    {
        context = Build(memories: [memoryFresh, memoryOld], recency: settings, now: now)!;
        Check(context.IndexOf("older memory", StringComparison.Ordinal) < context.IndexOf("fresh memory passage", StringComparison.Ordinal), "memory order ignores all modification dates and recency settings");
    }
    context = Build(memories: [memoryFresh, memoryOld], recency: decay, now: now)!;
    memoryOld.Item.Timestamp = now.AddDays(5);
    memoryFresh.Item.Timestamp = now.AddYears(-30);
    Check(Build(memories: [memoryFresh, memoryOld], recency: decay, now: now) == context, "memory modification times cannot alter context");

    var fusedOldMemory = Memory("fused-old.md", "older fused memory", .032f, timestamp: now.AddYears(-20));
    var fusedFreshMemory = Memory("fused-fresh.md", "fresh fused memory", .0318f, timestamp: now);
    fusedOldMemory.Legs = Legs(("gemma", .65f, 1), ("bge", .72f, 1));
    fusedFreshMemory.Legs = Legs(("gemma", .63f, 2), ("bge", .71f, 2));
    context = Build(both, memories: [fusedFreshMemory, fusedOldMemory], recency: decay, now: now)!;
    Check(context.IndexOf("older fused memory", StringComparison.Ordinal) < context.IndexOf("fresh fused memory", StringComparison.Ordinal), "fused memories never get freshness");

    // Fused hits: score is RRF, each leg carries its own model's cosine.
    context = Build(both, chats:
    [
        Chat("g", "gemma leg only", .0164f, legs: Legs(("gemma", .40f, 1))),
        Chat("b", "bge leg only", .0164f, legs: Legs(("bge", .65f, 1))),
    ])!;
    Check(context.Contains("gemma leg only") && !context.Contains("bge leg only"), "each leg is gated by its own floor");
    Check(ContextHook.Similarity(.0325f, Legs(("bge", .72f, 1), ("gemma", .30f, 2)), both) == .72f, "similarity is the best passing cosine");
    Check(ContextHook.Similarity(.0325f, Legs(("bge", .65f, 1), ("gemma", .40f, 2)), both) == .40f, "a leg under its own floor never supplies the similarity");
    context = Build(both, chats:
    [
        Chat("low", "lower rrf higher cosine", .0164f, legs: Legs(("bge", .95f, 1))),
        Chat("high", "higher rrf", .0325f, legs: Legs(("bge", .71f, 1), ("gemma", .39f, 1))),
    ])!;
    Check(context.IndexOf("higher rrf", StringComparison.Ordinal) < context.IndexOf("lower rrf higher cosine", StringComparison.Ordinal), "fused order follows RRF score");
    Check(context.Contains("  0.71  codex   high:17"), "fused line prints the gate cosine, not RRF");
    Check(Build(both, chats: [Chat("a", "unfused", .9f)]) is null, "fused selection needs legs");

    string? Build(ModelSpec[]? specs = null,
        (IndexItem Item, float Score, List<MemFind.Index.Leg>? Legs)[]? memories = null,
        (ChatIndex.Row Row, float Score, List<MemFind.Index.Leg>? Legs)[]? chats = null,
        RecencyOptions? recency = null, int k = 3, DateTimeOffset? now = null) =>
        ContextHook.BuildContext(evt, (memories ?? []).ToList(), (chats ?? []).ToList(), specs ?? gemma,
            recency ?? off, k, now);
    static bool Close(double a, double b) => Math.Abs(a - b) < 1e-9;

    using (var held = IndexLock.TryAcquire(root))
    {
        Check(held is not null, "initial store lock");
        Check(LockProbe(Path.Combine(root, ".") + Path.DirectorySeparatorChar) == "busy", "aliases cannot bypass writer lock");
        Check(LockProbe(Path.Combine(root, "other")) == "acquired", "independent stores can index concurrently");
        var alias = root + "-alias";
        if (!OperatingSystem.IsWindows())
        {
            Directory.CreateSymbolicLink(alias, root);
            try { Check(LockProbe(alias) == "busy", "symlink cannot bypass writer lock"); }
            finally { Directory.Delete(alias); }
        }
        if (Directory.Exists(root.ToUpperInvariant()))
            Check(LockProbe(root.ToUpperInvariant()) == "busy", "case aliases cannot bypass writer lock");
    }
    Check(LockProbe(root) == "acquired", "disposing releases writer lock");
    MetadataContracts(root);
    TranscriptContracts();

    // Store descriptions print whole, so long ones are what the character budget has to drop.
    var wide = Enumerable.Range(0, 3).Select(i => Memory($"w{i}.md", i + new string('x', 1400), .9f)).ToArray();
    context = Build(memories: wide, chats: [Chat("short", "short chat", .9f)])!;
    Check(context.Length <= ContextHook.MaxContext, "total context budget");
    lines = context.Split('\n');
    Check(lines.Length == 3 && lines[1].Contains("w0.md") && lines[2].Contains("short chat"), "lines over budget are dropped whole");

    var cache = Path.Combine(root, "cache");
    var gemmaCache = Path.Combine(cache, ModelSpec.Gemma.Name);
    Directory.CreateDirectory(gemmaCache);
    foreach (var file in ModelSpec.Gemma.Files)
        File.WriteAllText(Path.Combine(gemmaCache, file.File), "fixture");
    Check(!ContextHook.Prepared(cache, root, gemma), "missing indexes skip search");
    var index = MemFind.Index.PathFor(root, ModelSpec.Gemma);
    Directory.CreateDirectory(Path.GetDirectoryName(index)!);
    File.WriteAllText(index, "{}");
    Check(ContextHook.Prepared(cache, root, gemma), "prepared memory index");
    File.Delete(index);
    File.WriteAllText(ChatIndex.VecPath(root, ModelSpec.Gemma), "fixture");
    Check(ContextHook.Prepared(cache, root, gemma), "prepared chat index alone");
    foreach (var file in ModelSpec.Gemma.Files)
    {
        var path = Path.Combine(gemmaCache, file.File);
        File.WriteAllText(path, "");
        Check(!ContextHook.Prepared(cache, root, gemma), "empty model file cannot download");
        File.Delete(path);
        Check(!ContextHook.Prepared(cache, root, gemma), "missing model file cannot download");
        File.WriteAllText(path, "fixture");
    }
    Check(!ContextHook.Prepared(cache, root, both), "every selected model must have its files");
    var bgeCache = Path.Combine(cache, ModelSpec.Bge.Name);
    Directory.CreateDirectory(bgeCache);
    foreach (var file in ModelSpec.Bge.Files)
        File.WriteAllText(Path.Combine(bgeCache, file.File), "fixture");
    Check(!ContextHook.Prepared(cache, root, both), "every selected model must have its own index");
    File.WriteAllText(ChatIndex.VecPath(root, ModelSpec.Bge), "fixture");
    Check(ContextHook.Prepared(cache, root, both), "both models prepared");

    // Deadline: stdin held open blocks the work, so only the budget can end the process.
    var inputStart = Worker("--input", "1500");
    inputStart.RedirectStandardInput = true;
    using (var inputWorker = Process.Start(inputStart)!)
    {
        var inputElapsed = Stopwatch.StartNew();
        using var hostDeadline = new CancellationTokenSource(TimeSpan.FromSeconds(10));
        try { await inputWorker.WaitForExitAsync(hostDeadline.Token); }
        finally { if (!inputWorker.HasExited) inputWorker.Kill(entireProcessTree: true); }
        Check(inputWorker.ExitCode == 0 && inputElapsed.Elapsed < TimeSpan.FromSeconds(6), "open stdin obeys hook budget");
        Check(inputElapsed.Elapsed > TimeSpan.FromSeconds(.5), "budget is waited out, not skipped");
        Check(await inputWorker.StandardOutput.ReadToEndAsync() == "", "deadline emits no context");
    }
    using (var skipWorker = Process.Start(inputStart)!)
    {
        await skipWorker.StandardInput.WriteAsync("""{"hook_event_name":"UserPromptSubmit","prompt":"too short"}""");
        skipWorker.StandardInput.Close();
        using var hostDeadline = new CancellationTokenSource(TimeSpan.FromSeconds(10));
        try { await skipWorker.WaitForExitAsync(hostDeadline.Token); }
        finally { if (!skipWorker.HasExited) skipWorker.Kill(entireProcessTree: true); }
        Check(skipWorker.ExitCode == 0 && await skipWorker.StandardOutput.ReadToEndAsync() == "", "short prompt is skipped");
    }
    Console.WriteLine("PASS: hook format, gates, recency, metadata-only migration, transcript filtering, store locks, context budgets, offline preflight and deadline exit");
}
finally
{
    Directory.Delete(root, recursive: true);
}

static JsonElement Element(object value) => JsonSerializer.SerializeToElement(value);
static (IndexItem Item, float Score, List<MemFind.Index.Leg>? Legs) Memory(string file, string description, float score,
    int line = 0, DateTimeOffset? timestamp = null) =>
    (new IndexItem { File = file, Line = line, Title = "Title " + file, Description = description, Timestamp = timestamp }, score, null);
static (ChatIndex.Row Row, float Score, List<MemFind.Index.Leg>? Legs) Chat(string file, string text, float score,
    string session = "old", string ts = "2026-10-22T12:00:00Z", List<MemFind.Index.Leg>? legs = null, string source = "codex") =>
    (new ChatIndex.Row(source, score, file, 17, "uuid", 0, session, "/project", "user", ts, "project", session, ts, text),
        score, legs);
static List<MemFind.Index.Leg> Legs(params (string Model, float Score, int Rank)[] legs) =>
    legs.Select(leg => new MemFind.Index.Leg(leg.Model, leg.Score, leg.Rank)).ToList();
static void Check(bool condition, string reason)
{
    if (!condition)
        throw new Exception(reason);
}

static ProcessStartInfo Worker(params string[] args)
{
    var start = new ProcessStartInfo(Environment.ProcessPath!)
    {
        UseShellExecute = false, RedirectStandardOutput = true, RedirectStandardError = true,
    };
    if (Path.GetFileNameWithoutExtension(start.FileName) == "dotnet")
        start.ArgumentList.Add(System.Reflection.Assembly.GetExecutingAssembly().Location);
    foreach (var arg in args)
        start.ArgumentList.Add(arg);
    return start;
}

static string LockProbe(string path)
{
    using var process = Process.Start(Worker("--lock", path))!;
    var output = process.StandardOutput.ReadToEnd();
    var error = process.StandardError.ReadToEnd();
    process.WaitForExit();
    Check(process.ExitCode == 0, "lock probe failed: " + error);
    return output.Trim();
}

static void MetadataContracts(string root)
{
    var store = Path.Combine(root, "metadata-store");
    var codex = Path.Combine(root, "metadata-codex");
    Directory.CreateDirectory(store);
    Directory.CreateDirectory(codex);
    var plain = Path.Combine(store, "fact.md");
    var chunked = Path.Combine(codex, "MEMORY.md");
    File.WriteAllText(plain, "---\nname: Example\ndescription: A fact from curated memory.\n---\n");
    File.WriteAllText(chunked, "# First\nA first fact.\n# Second\nA second fact.\n");
    var original = Paths.CodexMemories;
    var downloads = Embedder.AllowDownloads;
    try
    {
        Paths.CodexMemories = codex;
        Embedder.AllowDownloads = false;
        var docs = Store.Parse(store, TranscriptSources.All);
        Check(docs.Count == 3 && docs.All(doc => doc.Timestamp is not null), "all memory passages have source dates");
        Check(Store.Parse(store, TranscriptSources.Claude).Count == 1, "claude-only drops Codex memories");
        var date = new DateTimeOffset(File.GetLastWriteTimeUtc(chunked));
        Check(docs.Where(doc => doc.File == chunked).All(doc => doc.Timestamp == date), "all chunks share the file mtime");
        // A legacy index with valid vectors but no ts must migrate without loading a model.
        var spec = ModelSpec.Gemma;
        var legacy = new IndexFile
        {
            Model = spec.Name, Dim = spec.Dim,
            Items = docs.Select(doc => new IndexItem
            {
                File = doc.File, Line = doc.Line, Theme = doc.Theme, Title = doc.Title,
                Description = doc.Description,
                TextHash = Convert.ToHexStringLower(SHA256.HashData(Encoding.UTF8.GetBytes(doc.Text))),
                Vec = Enumerable.Repeat(.125f, spec.Dim).ToArray(),
            }).ToList(),
        };
        var path = MemFind.Index.PathFor(store, spec);
        Directory.CreateDirectory(Path.GetDirectoryName(path)!);
        // Actually omit the new field, rather than only serializing it as null.
        File.WriteAllText(path, JsonSerializer.Serialize(legacy, Ctx.Default.IndexFile).Replace("\"ts\":null,", ""));
        Check(JsonSerializer.Deserialize(File.ReadAllText(path), Ctx.Default.IndexFile)!.Items.All(item => item.Timestamp is null), "legacy index remains readable");
        Check(MemFind.Index.Build(store, spec, false, TranscriptSources.All) == 0, "legacy metadata migration");
        var migrated = JsonSerializer.Deserialize(File.ReadAllText(path), Ctx.Default.IndexFile)!;
        Check(migrated.Items.All(item => item.Timestamp is not null && item.Vec.SequenceEqual(legacy.Items[0].Vec)), "migration updates dates and reuses vectors");
        var changed = date.AddDays(-14);
        File.SetLastWriteTimeUtc(chunked, changed.UtcDateTime);
        Check(MemFind.Index.Build(store, spec, false, TranscriptSources.All) == 0, "mtime-only reindex");
        var updated = JsonSerializer.Deserialize(File.ReadAllText(path), Ctx.Default.IndexFile)!;
        Check(updated.Items.Where(item => item.File == chunked).All(item => item.Timestamp == changed), "mtime-only edit persisted");
        Check(updated.Items.All(item => item.Vec.SequenceEqual(legacy.Items[0].Vec)), "mtime never changes embeddings");
        var bytes = File.ReadAllText(path);
        Check(MemFind.Index.Build(store, spec, false, TranscriptSources.All) == 0 && File.ReadAllText(path) == bytes, "unchanged metadata does not rewrite index");
        Check(MemFind.Index.Build(store, spec, false, TranscriptSources.Claude) == 0
            && JsonSerializer.Deserialize(File.ReadAllText(path), Ctx.Default.IndexFile)!.Items.Count == 1, "claude-only rewrites the memory index without Codex items");
    }
    finally
    {
        Paths.CodexMemories = original;
        Embedder.AllowDownloads = downloads;
    }
}

static void TranscriptContracts()
{
    var transcript = new Transcript(true, false, "session", "/project");
    const string timestamp = "2026-10-22T12:00:00Z";
    var message = JsonSerializer.SerializeToUtf8Bytes(new { type = "response_item", timestamp,
        payload = new { type = "message", role = "user", content = new[] { new { type = "input_text", text = "A real user message" } } } });
    Check(transcript.TryRead(message, 2, true, out var parsed) && parsed.Ts == timestamp, "chat uses original message timestamp");
    foreach (var type in new[] { "function_call", "function_call_output", "reasoning" })
    {
        var tool = JsonSerializer.SerializeToUtf8Bytes(new { type = "response_item", timestamp,
            payload = new { type, text = "Never index tool records" } });
        Check(!transcript.TryRead(tool, 3, true, out _), "Codex tool/reasoning records excluded");
    }
    var claude = JsonSerializer.SerializeToUtf8Bytes(new { type = "user", timestamp, message = new { role = "user",
        content = new[] { new { type = "tool_result", text = "Never index a tool result" } } } });
    Check(!Chats.TryReadKept(claude, true, out _), "Claude tool result excluded");
}
