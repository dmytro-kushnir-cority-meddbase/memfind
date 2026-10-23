using System.Diagnostics;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using MemFind;

if (args.Length == 2 && args[0] == "--sleep")
{
    File.WriteAllText(args[1], Environment.ProcessId.ToString());
    await Task.Delay(TimeSpan.FromSeconds(30));
    return;
}
if (args.Length == 1 && args[0] == "--input")
{
    await ContextHook.RunAsync(Path.GetTempPath());
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
    var evt = Element(new { session_id = "current", transcript_path = Path.Combine(root, "current.jsonl") });
    var rows = new[]
    {
        Hit("a", "below", .37999),
        Hit("b", "boundary", .38),
        Hit("c", "highest", .9),
        Hit("d", "current session", .99, session: "current"),
        Hit(Path.Combine(root, "current.jsonl"), "current transcript", .99),
        Hit("c", "another chunk", .8),
        Hit("e", " HIGHEST  ", .8),
    };
    var context = ContextHook.BuildContext(evt, string.Join('\n', rows), root)!;
    Check(context.StartsWith("<memfind_context>\n") && context.EndsWith("\n</memfind_context>"), "context envelope");
    Check(context.Contains("EmbeddingGemma") && context.Contains("untrusted reference data"), "source label");
    Check(!context.Contains("below") && context.Contains("boundary"), "Gemma score floor");
    Check(context.IndexOf("highest", StringComparison.Ordinal) < context.IndexOf("boundary", StringComparison.Ordinal), "score order");
    Check(!context.Contains("current session") && !context.Contains("current transcript"), "exclude current source");
    Check(!context.Contains("another chunk") && !context.Contains("HIGHEST"), "deduplication");
    Check(context.Contains(Path.Combine(root, "b")), "relative anchor uses selected store");
    Check(ContextHook.BuildContext(evt, Hit("a", "low", .1), root) is null, "no irrelevant context");

    var now = DateTimeOffset.Parse("2026-10-22T12:00:00Z");
    var fresh = now.ToString("O");
    var old = now.AddDays(-365).ToString("O");
    var dated = string.Join('\n', new[]
    {
        Hit("old", "older near match", .65, ts: old),
        Hit("fresh", "fresh near match", .63, ts: fresh),
        Hit("strong", "strong older match", .8, ts: old),
        Hit("irrelevant", "fresh irrelevant", .379, ts: fresh),
    });
    context = ContextHook.BuildContext(evt, dated, root, now)!;
    Check(context.IndexOf("strong older match", StringComparison.Ordinal) < context.IndexOf("fresh near match", StringComparison.Ordinal)
        && context.IndexOf("fresh near match", StringComparison.Ordinal) < context.IndexOf("older near match", StringComparison.Ordinal), "bounded freshness reranking");
    Check(!context.Contains("fresh irrelevant"), "freshness never bypasses similarity floor");
    var header = Header(Hit("a", "timestamp example", .6, ts: fresh));
    Check(header.GetProperty("ts").GetDateTimeOffset() == now && Rank(header) == .66, "timestamp and raw/ranked scores exposed");
    Check(header.GetProperty("score").GetDouble() == .6, "raw similarity is preserved");
    Check(Rank(Header(Hit("a", "no date", .6))) == .6, "missing date has no bonus");
    Check(Rank(Header(Hit("a", "bad date", .6, ts: "invalid"))) == .6, "invalid date has no bonus");
    Check(Rank(Header(Hit("a", "future date", .6, ts: now.AddDays(3).ToString("O")))) == .66, "future date bonus is clamped");
    Check(Rank(Header(Hit("a", "week old", .6, ts: now.AddDays(-7).ToString("O")))) == .63, "seven day half life");

    JsonElement Header(string hit)
    {
        var built = ContextHook.BuildContext(evt, hit, root, now)!;
        return JsonDocument.Parse(built.Split('\n').Single(line => line.StartsWith('{'))).RootElement.Clone();
    }
    static double Rank(JsonElement header) => header.GetProperty("rank_score").GetDouble();

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
    JsonLines.FullPrecision = true;
    Check(JsonLines.Score(.37999f) < ContextHook.MinScore, "hook worker preserves below-floor precision");
    JsonLines.FullPrecision = false;
    MetadataContracts(root);
    TranscriptContracts();

    var many = string.Join('\n', Enumerable.Range(0, 10).Select(i => Hit("source" + i, i + new string('x', 900), .9)));
    context = ContextHook.BuildContext(evt, many, root)!;
    Check(context.Length <= ContextHook.MaxContext, "total context budget");
    Check(context.Split("\nexcerpt: ").Length - 1 <= 4, "hit budget");
    foreach (var line in context.Split('\n').Where(line => line.StartsWith("excerpt: ")))
        Check(JsonSerializer.Deserialize<string>(line[9..])!.Length <= 500, "excerpt budget");

    var cache = Path.Combine(root, "cache");
    Directory.CreateDirectory(cache);
    foreach (var file in ModelSpec.Gemma.Files)
        File.WriteAllText(Path.Combine(cache, file.File), "fixture");
    Check(!ContextHook.Prepared(cache, root), "missing indexes skip search");
    var index = MemFind.Index.PathFor(root, ModelSpec.Gemma);
    Directory.CreateDirectory(Path.GetDirectoryName(index)!);
    File.WriteAllText(index, "{}");
    Check(ContextHook.Prepared(cache, root), "prepared memory index");
    File.Delete(index);
    File.WriteAllText(ChatIndex.VecPath(root, ModelSpec.Gemma), "fixture");
    Check(ContextHook.Prepared(cache, root), "prepared chat index alone");
    foreach (var file in ModelSpec.Gemma.Files)
    {
        var path = Path.Combine(cache, file.File);
        File.WriteAllText(path, "");
        Check(!ContextHook.Prepared(cache, root), "empty model file cannot download");
        File.Delete(path);
        Check(!ContextHook.Prepared(cache, root), "missing model file cannot download");
        File.WriteAllText(path, "fixture");
    }

    // Hold stdin open: Console's synchronized reader must not defeat the deadline.
    var inputStart = Worker("--input");
    inputStart.RedirectStandardInput = true;
    using (var inputWorker = Process.Start(inputStart)!)
    {
        var inputElapsed = Stopwatch.StartNew();
        using var hostDeadline = new CancellationTokenSource(TimeSpan.FromSeconds(10));
        try { await inputWorker.WaitForExitAsync(hostDeadline.Token); }
        finally { if (!inputWorker.HasExited) inputWorker.Kill(entireProcessTree: true); }
        Check(inputWorker.ExitCode == 0 && inputElapsed.Elapsed < TimeSpan.FromSeconds(10), "open stdin obeys hook budget");
        Check(await inputWorker.StandardOutput.ReadToEndAsync() == "", "input timeout emits no context");
    }

    // Real subprocess cancellation: no mocked timeout and no Python interpreter.
    var pidFile = Path.Combine(root, "slow.pid");
    var start = Worker("--sleep", pidFile);
    using var cancellation = new CancellationTokenSource(ContextHook.Budget);
    var elapsed = Stopwatch.StartNew();
    try
    {
        await ContextHook.ReadProcessAsync(start, cancellation.Token);
        throw new Exception("Expected timeout");
    }
    catch (OperationCanceledException) { }
    Check(elapsed.Elapsed < TimeSpan.FromSeconds(10), "cleanup fits host timeout");
    Check(File.Exists(pidFile), "slow worker actually started");
    try
    {
        using var child = Process.GetProcessById(int.Parse(File.ReadAllText(pidFile)));
        Check(child.HasExited, "timed-out worker terminated");
    }
    catch (ArgumentException) { }
    Check(ContextHook.Budget.TotalSeconds == 9.5, "cleanup fits ten-second host budget");
    Console.WriteLine("PASS: recency, metadata-only migration, transcript filtering, store locks, context budgets, offline preflight and real process termination");
}
finally
{
    Directory.Delete(root, recursive: true);
}

static JsonElement Element(object value) => JsonSerializer.SerializeToElement(value);
static string Hit(string file, string text, double score, string session = "old", string? ts = null) =>
    JsonSerializer.Serialize(new { file, text, score, session, ts, line = 17, source = "codex" });
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
        var docs = Store.Parse(store);
        Check(docs.Count == 3 && docs.All(doc => doc.Timestamp is not null), "all memory passages have source dates");
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
        Check(MemFind.Index.Build(store, spec, false) == 0, "legacy metadata migration");
        var migrated = JsonSerializer.Deserialize(File.ReadAllText(path), Ctx.Default.IndexFile)!;
        Check(migrated.Items.All(item => item.Timestamp is not null && item.Vec.SequenceEqual(legacy.Items[0].Vec)), "migration updates dates and reuses vectors");
        var changed = date.AddDays(-14);
        File.SetLastWriteTimeUtc(chunked, changed.UtcDateTime);
        Check(MemFind.Index.Build(store, spec, false) == 0, "mtime-only reindex");
        var updated = JsonSerializer.Deserialize(File.ReadAllText(path), Ctx.Default.IndexFile)!;
        Check(updated.Items.Where(item => item.File == chunked).All(item => item.Timestamp == changed), "mtime-only edit persisted");
        Check(updated.Items.All(item => item.Vec.SequenceEqual(legacy.Items[0].Vec)), "mtime never changes embeddings");
        var bytes = File.ReadAllText(path);
        Check(MemFind.Index.Build(store, spec, false) == 0 && File.ReadAllText(path) == bytes, "unchanged metadata does not rewrite index");
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
