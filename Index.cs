using System.Diagnostics;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace MemFind;

internal sealed class IndexItem
{
    [JsonPropertyName("file")]
    public string File { get; set; } = "";

    [JsonPropertyName("theme")]
    public string Theme { get; set; } = "";

    [JsonPropertyName("title")]
    public string Title { get; set; } = "";

    [JsonPropertyName("description")]
    public string Description { get; set; } = "";

    [JsonPropertyName("textHash")]
    public string TextHash { get; set; } = "";

    [JsonPropertyName("vec")]
    public float[] Vec { get; set; } = [];
}

internal sealed class IndexFile
{
    [JsonPropertyName("model")]
    public string Model { get; set; } = "";

    [JsonPropertyName("dim")]
    public int Dim { get; set; }

    [JsonPropertyName("builtAt")]
    public string BuiltAt { get; set; } = "";

    [JsonPropertyName("items")]
    public List<IndexItem> Items { get; set; } = [];
}

internal static class Index
{
    /// <summary>Rank-fusion constant; the models' cosines are on different scales, so only ranks fuse.</summary>
    public const float RrfK = 60f;

    public static string PathFor(string root, ModelSpec spec) =>
        System.IO.Path.Combine(root, ".memfind", $"index{spec.Suffix}.json");

    public static int Build(string root, ModelSpec spec, bool gpu)
    {
        if (!Directory.Exists(root))
        {
            Console.Error.WriteLine($"store not found: {root}");
            return 1;
        }

        var sw = Stopwatch.StartNew();
        var docs = Store.Parse(root);

        var path = PathFor(root, spec);
        var old = LoadExisting(path);
        var reusable = BuildReusable(old, spec);

        var items = new List<IndexItem>(docs.Count);
        var pending = new List<(int Index, string Text)>();

        foreach (var doc in docs)
        {
            var hash = HashText(doc.Text);
            var item = new IndexItem
            {
                File = doc.File,
                Theme = doc.Theme,
                Title = doc.Title,
                Description = doc.Description,
                TextHash = hash,
            };

            if (reusable is not null && reusable.TryGetValue((doc.File, hash), out var reused))
                item.Vec = reused.Vec;
            else
                pending.Add((items.Count, doc.Text));

            items.Add(item);
        }

        if (pending.Count > 0)
        {
            using var embedder = Embedder.Load(spec, gpu);
            foreach (var (i, text) in pending)
                items[i].Vec = embedder.EmbedDocument(text);
        }

        var dir = System.IO.Path.Combine(root, ".memfind");
        Directory.CreateDirectory(dir);
        File.WriteAllText(System.IO.Path.Combine(dir, ".gitignore"), "*\n");

        if (pending.Count == 0 && old is not null && SameItems(old.Items, items))
        {
            sw.Stop();
            Console.Error.WriteLine($"memories unchanged ({items.Count}) -> {path}");
            return 0;
        }

        var index = new IndexFile
        {
            Model = spec.Name,
            Dim = spec.Dim,
            BuiltAt = DateTime.UtcNow.ToString("o"),
            Items = items,
        };

        var tmp = path + ".tmp";
        File.WriteAllText(tmp, JsonSerializer.Serialize(index, Ctx.Default.IndexFile));
        File.Move(tmp, path, overwrite: true);

        sw.Stop();
        Console.Error.WriteLine(
            $"indexed {items.Count} memories ({pending.Count} embedded, {items.Count - pending.Count} reused) in {sw.Elapsed.TotalSeconds:F1} s -> {path}"
        );
        return 0;
    }

    private static IndexFile? LoadExisting(string path)
    {
        if (!File.Exists(path))
            return null;
        try
        {
            return JsonSerializer.Deserialize(File.ReadAllText(path), Ctx.Default.IndexFile);
        }
        catch
        {
            return null;
        }
    }

    /// <summary>Old items keyed by (File, TextHash); items missing textHash (old format) never match.</summary>
    private static Dictionary<(string File, string Hash), IndexItem>? BuildReusable(
        IndexFile? old,
        ModelSpec spec
    )
    {
        if (old is null || old.Model != spec.Name || old.Dim != spec.Dim)
            return null;

        var map = new Dictionary<(string, string), IndexItem>();
        foreach (var item in old.Items)
            if (item.TextHash.Length > 0)
                map.TryAdd((item.File, item.TextHash), item);
        return map;
    }

    private static bool SameItems(List<IndexItem> a, List<IndexItem> b)
    {
        if (a.Count != b.Count)
            return false;
        for (var i = 0; i < a.Count; i++)
        {
            var x = a[i];
            var y = b[i];
            if (
                x.File != y.File
                || x.Theme != y.Theme
                || x.Title != y.Title
                || x.Description != y.Description
                || x.TextHash != y.TextHash
            )
                return false;
        }
        return true;
    }

    private static string HashText(string text) =>
        Convert.ToHexStringLower(SHA256.HashData(Encoding.UTF8.GetBytes(text)));

    private static List<(IndexItem Item, float Score)>? Rank(
        string root,
        ModelSpec spec,
        string query,
        int k,
        Dictionary<ModelSpec, Task<Embedder>> embedders
    )
    {
        var path = PathFor(root, spec);
        if (File.Exists(path))
            Embedder.Preload(embedders, spec);
        var index = File.Exists(path)
            ? JsonSerializer.Deserialize(File.ReadAllText(path), Ctx.Default.IndexFile)
            : null;

        if (index is null || index.Items.Count == 0)
        {
            Console.Error.WriteLine(
                spec.Suffix.Length == 0
                    ? "no index — run: memfind index"
                    : $"no {spec.Key} index — run: memfind index --model {spec.Key}"
            );
            return null;
        }

        var embedder = Embedder.ForQuery(embedders, ModelSpec.ByName(index.Model) ?? spec);
        var q = embedder.EmbedQuery(query);

        return index
            .Items.Select(item => (Item: item, Score: Dot(q, item.Vec)))
            .OrderByDescending(h => h.Score)
            .Take(k)
            .ToList();
    }

    public static int Search(
        string root,
        ModelSpec[] specs,
        string query,
        int k,
        bool json,
        Dictionary<ModelSpec, Task<Embedder>> embedders
    )
    {
        if (specs.Length > 1)
            return SearchFused(root, specs, query, k, json, embedders);

        var hits = Rank(root, specs[0], query, k, embedders);
        if (hits is null)
            return 1;

        if (json)
        {
            using var output = new JsonLines();
            foreach (var (item, score) in hits)
            {
                var writer = output.Writer;
                writer.WriteStartObject();
                writer.WriteString("source"u8, "memory");
                writer.WriteNumber("score"u8, Math.Round(score, 4));
                writer.WriteString("file"u8, item.File);
                writer.WriteString("theme"u8, item.Theme);
                writer.WriteString("title"u8, item.Title);
                writer.WriteString("description"u8, item.Description);
                writer.WriteEndObject();
                output.Flush();
            }

            return 0;
        }

        var themeWidth = hits.Count == 0 ? 0 : hits.Max(h => h.Item.Theme.Length);
        var fileWidth = hits.Count == 0 ? 0 : hits.Max(h => h.Item.File.Length);

        foreach (var (item, score) in hits)
        {
            Console.WriteLine(
                $"{score:F3}  memory  {item.Theme.PadRight(themeWidth)}   {item.File.PadRight(fileWidth)}   {item.Title}"
            );
            Console.WriteLine($"       {Truncate(item.Description, 160)}");
        }

        return 0;
    }

    private static int SearchFused(
        string root,
        ModelSpec[] specs,
        string query,
        int k,
        bool json,
        Dictionary<ModelSpec, Task<Embedder>> embedders
    )
    {
        var fused = new Dictionary<string, (IndexItem Item, float Score, List<Leg> Legs)>();
        var legs = 0;

        foreach (var spec in specs)
        {
            var hits = Rank(root, spec, query, 3 * k, embedders);
            if (hits is null)
                continue;
            legs++;

            for (var rank = 1; rank <= hits.Count; rank++)
            {
                var (item, score) = hits[rank - 1];
                if (!fused.TryGetValue(item.File, out var entry))
                    entry = (item, 0f, []);
                entry.Legs.Add(new Leg(spec.Key, score, rank));
                fused[item.File] = (entry.Item, entry.Score + 1f / (RrfK + rank), entry.Legs);
            }
        }

        if (legs == 0)
            return 1;

        var rows = fused
            .Values.OrderByDescending(h => h.Score)
            .ThenBy(h => h.Item.File, StringComparer.Ordinal)
            .Take(k)
            .ToList();

        if (json)
        {
            using var output = new JsonLines();
            foreach (var (item, score, hitLegs) in rows)
            {
                var writer = output.Writer;
                writer.WriteStartObject();
                writer.WriteString("source"u8, "memory");
                writer.WriteNumber("score"u8, Math.Round(score, 4));
                writer.WriteString("file"u8, item.File);
                writer.WriteString("theme"u8, item.Theme);
                writer.WriteString("title"u8, item.Title);
                writer.WriteString("description"u8, item.Description);
                WriteLegs(writer, hitLegs);
                writer.WriteEndObject();
                output.Flush();
            }

            return 0;
        }

        var themeWidth = rows.Count == 0 ? 0 : rows.Max(h => h.Item.Theme.Length);
        var fileWidth = rows.Count == 0 ? 0 : rows.Max(h => h.Item.File.Length);

        foreach (var (item, score, hitLegs) in rows)
        {
            Console.WriteLine(
                $"{score:F3}  memory  {item.Theme.PadRight(themeWidth)}   {item.File.PadRight(fileWidth)}   {item.Title}   [{Describe(hitLegs)}]"
            );
            Console.WriteLine($"       {Truncate(item.Description, 160)}");
        }

        return 0;
    }

    internal readonly record struct Leg(string Model, float Score, int Rank);

    internal static string Describe(List<Leg> legs) =>
        string.Join(' ', legs.Select(l => $"{l.Model} #{l.Rank} {l.Score:F3}"));

    internal static void WriteLegs(Utf8JsonWriter writer, List<Leg> legs)
    {
        writer.WriteStartArray("models"u8);
        foreach (var leg in legs)
        {
            writer.WriteStartObject();
            writer.WriteString("model"u8, leg.Model);
            writer.WriteNumber("score"u8, Math.Round(leg.Score, 4));
            writer.WriteNumber("rank"u8, leg.Rank);
            writer.WriteEndObject();
        }
        writer.WriteEndArray();
    }

    private static float Dot(float[] a, float[] b)
    {
        var n = Math.Min(a.Length, b.Length);
        float sum = 0;
        for (var i = 0; i < n; i++)
            sum += a[i] * b[i];
        return sum;
    }

    private static string Truncate(string s, int max)
    {
        s = s.Replace('\n', ' ').Replace('\r', ' ').Trim();
        return s.Length <= max ? s : s[..max] + "...";
    }
}
