using System.Diagnostics;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace MemFind;

internal sealed class IndexItem
{
    [JsonPropertyName("file")] public string File { get; set; } = "";
    [JsonPropertyName("theme")] public string Theme { get; set; } = "";
    [JsonPropertyName("title")] public string Title { get; set; } = "";
    [JsonPropertyName("description")] public string Description { get; set; } = "";
    [JsonPropertyName("vec")] public float[] Vec { get; set; } = [];
}

internal sealed class IndexFile
{
    [JsonPropertyName("model")] public string Model { get; set; } = Embedder.ModelName;
    [JsonPropertyName("dim")] public int Dim { get; set; } = Embedder.Dim;
    [JsonPropertyName("builtAt")] public string BuiltAt { get; set; } = "";
    [JsonPropertyName("items")] public List<IndexItem> Items { get; set; } = [];
}

internal static class Index
{
    private static readonly JsonSerializerOptions Json = new() { WriteIndented = false };

    public static string PathFor(string root) => System.IO.Path.Combine(root, ".memfind", "index.json");

    public static int Build(string root)
    {
        if (!Directory.Exists(root))
        {
            Console.Error.WriteLine($"store not found: {root}");
            return 1;
        }

        var sw = Stopwatch.StartNew();
        var docs = Store.Parse(root);

        using var embedder = Embedder.Load();

        var index = new IndexFile { BuiltAt = DateTime.UtcNow.ToString("o") };
        foreach (var doc in docs)
        {
            index.Items.Add(new IndexItem
            {
                File = doc.File,
                Theme = doc.Theme,
                Title = doc.Title,
                Description = doc.Description,
                Vec = embedder.EmbedDocument(doc.Text),
            });
        }

        var dir = System.IO.Path.Combine(root, ".memfind");
        Directory.CreateDirectory(dir);
        File.WriteAllText(System.IO.Path.Combine(dir, ".gitignore"), "*\n");
        File.WriteAllText(PathFor(root), JsonSerializer.Serialize(index, Json));

        sw.Stop();
        Console.Error.WriteLine($"indexed {index.Items.Count} memories in {sw.Elapsed.TotalSeconds:F1} s -> {PathFor(root)}");
        return 0;
    }

    public static int Search(string root, string query, int k)
    {
        var path = PathFor(root);
        if (!File.Exists(path))
        {
            Console.Error.WriteLine("no index — run: memfind index");
            return 1;
        }

        var index = JsonSerializer.Deserialize<IndexFile>(File.ReadAllText(path), Json);
        if (index is null || index.Items.Count == 0)
        {
            Console.Error.WriteLine("no index — run: memfind index");
            return 1;
        }

        using var embedder = Embedder.Load();
        var q = embedder.EmbedQuery(query);

        var hits = index.Items
            .Select(item => (Item: item, Score: Dot(q, item.Vec)))
            .OrderByDescending(h => h.Score)
            .Take(k)
            .ToList();

        var themeWidth = hits.Count == 0 ? 0 : hits.Max(h => h.Item.Theme.Length);
        var fileWidth = hits.Count == 0 ? 0 : hits.Max(h => h.Item.File.Length);

        foreach (var (item, score) in hits)
        {
            Console.WriteLine($"{score:F3}  {item.Theme.PadRight(themeWidth)}   {item.File.PadRight(fileWidth)}   {item.Title}");
            Console.WriteLine($"       {Truncate(item.Description, 160)}");
        }

        return 0;
    }

    private static float Dot(float[] a, float[] b)
    {
        var n = Math.Min(a.Length, b.Length);
        float sum = 0;
        for (var i = 0; i < n; i++) sum += a[i] * b[i];
        return sum;
    }

    private static string Truncate(string s, int max)
    {
        s = s.Replace('\n', ' ').Replace('\r', ' ').Trim();
        return s.Length <= max ? s : s[..max] + "...";
    }
}
