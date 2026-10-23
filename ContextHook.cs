using System.Diagnostics;
using System.Globalization;
using System.Runtime.InteropServices;
using System.Text;
using System.Text.Encodings.Web;
using System.Text.Json;

namespace MemFind;

/// <summary>Additive RRF bonus for chats; the power-law bonus halves at HalfLifeDays.</summary>
internal sealed record RecencyOptions
{
    public double Boost { get; init; } = .001;
    public double HalfLifeDays { get; init; } = 21;
}

/// <summary>Advisory prompt context. Missing data, errors and timeout leave the prompt unchanged.</summary>
internal static partial class ContextHook
{
    internal const int MaxContext = 3000;
    // Candidates per source; the gate, exclusions and dedup then keep k of them.
    private const int Candidates = 64;
    internal const string Header = "memfind: memories, past chats and subagent chats matching this prompt - Read the "
        + "relevant ones before searching the filesystem; source is the second column (memory = curated store, "
        + "chat = main session, agent = subagent transcript whose \"user\" is the brief); chat and agent hits are "
        + "speculative (`memfind show <path:line> -c 3` for context):";

    public static int Run(string store, ModelSpec[] specs, RecencyOptions recency, int k, int budgetMs, Stream stdout)
    {
        // The host's timeout starts before the runtime does; charge startup against the budget.
        var remaining = TimeSpan.FromMilliseconds(budgetMs) - (DateTime.Now - Process.GetCurrentProcess().StartTime);
        Console.SetError(TextWriter.Null);
        var work = Task.Run(() => Context(store, specs, recency, k));
        try
        {
            if (remaining > TimeSpan.Zero && work.Wait(remaining) && work.Result is { } context)
            {
                using var writer = new Utf8JsonWriter(stdout,
                    new JsonWriterOptions { Encoder = JavaScriptEncoder.UnsafeRelaxedJsonEscaping });
                writer.WriteStartObject();
                writer.WriteStartObject("hookSpecificOutput");
                writer.WriteString("hookEventName", "UserPromptSubmit");
                writer.WriteString("additionalContext", context);
                writer.WriteEndObject();
                writer.WriteEndObject();
                writer.Flush();
                stdout.Flush();
            }
        }
        catch (Exception)
        {
            // Retrieval is optional. Never emit a partial context or block the user on failure.
        }
        if (!work.IsCompleted)
        {
            // A model load may still be inside ONNX Runtime: skip runtime shutdown and unload entirely.
            try
            {
                if (OperatingSystem.IsWindows())
                    TerminateProcess(GetCurrentProcess(), 0);
                else
                    _exit(0);
            }
            catch (Exception)
            {
                // A missing libc binding must still end the process with exit code 0.
            }
            Environment.Exit(0);
        }
        return 0;
    }

    [LibraryImport("kernel32.dll")]
    private static partial nint GetCurrentProcess();

    [LibraryImport("kernel32.dll")]
    private static partial int TerminateProcess(nint process, uint exitCode);

    [LibraryImport("libc")]
    private static partial void _exit(int status);

    private static string? Context(string store, ModelSpec[] specs, RecencyOptions recency, int k)
    {
        // Runs on the work task, so an open stdin is bounded by the caller's deadline.
        string input;
        using (var reader = new StreamReader(Console.OpenStandardInput(), Encoding.UTF8))
            input = reader.ReadToEnd();
        using var document = JsonDocument.Parse(input);
        var evt = document.RootElement;
        var prompt = Get(evt, "prompt");
        if (Get(evt, "hook_event_name") != "UserPromptSubmit"
            || prompt.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries).Length < 3)
            return null;

        var cache = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
            "memfind");
        if (!Prepared(cache, store, specs))
            return null;

        Embedder.AllowDownloads = false;
        var query = prompt[..Math.Min(prompt.Length, 2000)];
        var embedders = new Dictionary<ModelSpec, Task<Embedder>>();
        try
        {
            // Prepared guarantees an index per model, so every load can start now and overlap.
            foreach (var spec in specs)
                Embedder.Preload(embedders, spec);
            var memories = Index.Hits(store, specs, query, Candidates, embedders) ?? [];
            var chats = ChatIndex.Hits(store, specs, query, Candidates, embedders) ?? [];
            return BuildContext(evt, memories, chats, specs, recency, k);
        }
        finally
        {
            Embedder.DisposeAll(embedders);
        }
    }

    internal static bool Prepared(string cache, string store, ModelSpec[] specs) =>
        specs.All(spec => spec.Files.All(file => Ready(Path.Combine(cache, spec.Name, file.File)))
            && (Ready(Index.PathFor(store, spec)) || Ready(ChatIndex.VecPath(store, spec))));

    private static bool Ready(string file) => File.Exists(file) && new FileInfo(file).Length > 0;

    // The gate is each model's cosine against its own floor; RRF scores are only for ordering.
    internal static double? Similarity(float score, List<Index.Leg>? legs, ModelSpec[] specs)
    {
        if (legs is null)
            return specs.Length == 1 && score >= specs[0].Floor ? score : null;
        double? best = null;
        foreach (var leg in legs)
        {
            var spec = specs.FirstOrDefault(m => m.Key == leg.Model);
            if (spec is not null && leg.Score >= spec.Floor && (best is null || leg.Score > best))
                best = leg.Score;
        }
        return best;
    }

    internal static double Rank(double score, DateTimeOffset? timestamp, RecencyOptions recency, DateTimeOffset now)
    {
        var ageDays = timestamp is { } ts ? Math.Max(0, (now - ts).TotalDays) : double.PositiveInfinity;
        return score + recency.Boost / Math.Sqrt(1 + 3 * ageDays / recency.HalfLifeDays);
    }

    internal static string? BuildContext(JsonElement evt,
        List<(IndexItem Item, float Score, List<Index.Leg>? Legs)> memories,
        List<(ChatIndex.Row Row, float Score, List<Index.Leg>? Legs)> chats,
        ModelSpec[] specs, RecencyOptions recency, int k, DateTimeOffset? now = null)
    {
        var clock = now ?? DateTimeOffset.UtcNow;
        var session = Get(evt, "session_id");
        var transcript = Get(evt, "transcript_path");
        var bodies = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        var lines = new List<string>();

        var count = 0;
        foreach (var (item, similarity) in Gated(memories, null, specs, recency, clock))
        {
            if (count == k)
                break;
            if (!bodies.Add(Collapse(item.Description.Length > 0 ? item.Description : item.Title)))
                continue;
            // Codex memory passages are chunks of a longer file: anchor and shorten them like chat text.
            lines.Add(item.Line > 0
                ? string.Create(CultureInfo.InvariantCulture,
                    $"  {similarity:0.00}  memory  {item.File}:{item.Line}  {item.Title} - {Shorten(item.Description)}")
                : string.Create(CultureInfo.InvariantCulture,
                    $"  {similarity:0.00}  memory  {item.File}  {item.Title} - {item.Description}"));
            count++;
        }

        count = 0;
        foreach (var (row, similarity) in Gated(chats, row => Timestamp(row.Ts), specs, recency, clock))
        {
            if (count == k)
                break;
            if ((session.Length > 0 && row.Session == session)
                || (transcript.Length > 0 && Paths.Comparer.Equals(Path.GetFullPath(row.Path), Path.GetFullPath(transcript)))
                || !bodies.Add(Collapse(row.Text)))
                continue;
            var date = Timestamp(row.Ts)?.ToString("yyyy-MM-dd", CultureInfo.InvariantCulture) ?? row.Ts;
            lines.Add(string.Create(CultureInfo.InvariantCulture,
                $"  {similarity:0.00}  {row.Source,-6}  {row.Path}:{row.Line}  {date} {row.Role} [{row.Cwd}] - {Shorten(row.Text)}"));
            count++;
        }

        var context = new StringBuilder(Header);
        var kept = 0;
        foreach (var line in lines)
        {
            if (context.Length + 1 + line.Length > MaxContext)
                continue;
            context.Append('\n').Append(line);
            kept++;
        }
        return kept == 0 ? null : context.ToString();
    }

    private static IEnumerable<(T Hit, double Similarity)> Gated<T>(List<(T Hit, float Score, List<Index.Leg>? Legs)> hits,
        Func<T, DateTimeOffset?>? timestamp, ModelSpec[] specs, RecencyOptions recency, DateTimeOffset now) =>
        hits.OrderByDescending(h => h.Score)
            // Single-model chat ranks use one RRF leg, assigned before the relevance gate.
            .Select((h, i) => (h.Hit, Similarity: Similarity(h.Score, h.Legs, specs),
                Score: timestamp is not null && h.Legs is null ? 1d / (Index.RrfK + i + 1) : h.Score))
            .Where(h => h.Similarity is not null)
            .OrderByDescending(h => timestamp is null ? h.Score : Rank(h.Score, timestamp(h.Hit), recency, now))
            .Select(h => (h.Hit, h.Similarity!.Value));

    private static DateTimeOffset? Timestamp(string ts) =>
        DateTimeOffset.TryParse(ts, CultureInfo.InvariantCulture,
            DateTimeStyles.AssumeUniversal | DateTimeStyles.AdjustToUniversal, out var parsed) ? parsed : null;

    private static string Collapse(string text) =>
        string.Join(' ', text.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries));

    private static string Shorten(string text)
    {
        text = Collapse(text);
        // A lone high surrogate would make the JSON writer throw and drop the whole context.
        return text.Length > 160 ? text[..(char.IsHighSurrogate(text[159]) ? 159 : 160)] + "..." : text;
    }

    private static string Get(JsonElement element, string name) =>
        element.ValueKind == JsonValueKind.Object && element.TryGetProperty(name, out var value)
        && value.ValueKind == JsonValueKind.String ? value.GetString() ?? "" : "";
}
