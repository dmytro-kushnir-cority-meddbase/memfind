using System.Text.RegularExpressions;

namespace MemFind;

internal sealed record MemoryDoc(
    string File,
    string Theme,
    string Title,
    string Description,
    string Hook,
    string Type
)
{
    public string Text
    {
        get
        {
            var parts = new List<string>(3);
            if (!string.IsNullOrWhiteSpace(Title))
                parts.Add(Title.Trim());
            if (!string.IsNullOrWhiteSpace(Description))
                parts.Add(Description.Trim());
            if (!string.IsNullOrWhiteSpace(Hook))
                parts.Add(Hook.Trim());
            return string.Join(". ", parts);
        }
    }
}

internal static class Store
{
    public const string DefaultRoot = @"C:\Git\docs\memory";

    public static List<MemoryDoc> Parse(string root)
    {
        var placement = BuildPlacement(root);
        var docs = new List<MemoryDoc>();

        foreach (
            var path in Directory
                .GetFiles(root, "*.md", SearchOption.TopDirectoryOnly)
                .OrderBy(p => p, StringComparer.OrdinalIgnoreCase)
        )
        {
            var fileName = Path.GetFileName(path);
            if (string.Equals(fileName, "MEMORY.md", StringComparison.OrdinalIgnoreCase))
                continue;

            var (name, description, type) = ReadFrontMatter(path);
            var title = name;
            var theme = "(orphan)";
            var hook = "";

            if (placement.TryGetValue(fileName, out var p))
            {
                theme = p.Theme;
                if (!string.IsNullOrWhiteSpace(p.Title))
                    title = p.Title;
                hook = p.Hook;
            }

            if (string.IsNullOrWhiteSpace(title))
                title = Path.GetFileNameWithoutExtension(fileName);

            docs.Add(new MemoryDoc(fileName, theme, title, description, hook, type));
        }

        return docs;
    }

    private sealed record Placement(string Theme, string Title, string Hook);

    private static Dictionary<string, Placement> BuildPlacement(string root)
    {
        var map = new Dictionary<string, Placement>(StringComparer.OrdinalIgnoreCase);

        var indexDir = Path.Combine(root, "index");
        if (Directory.Exists(indexDir))
        {
            foreach (
                var indexFile in Directory.GetFiles(indexDir, "*.md", SearchOption.TopDirectoryOnly)
            )
            {
                var theme = Path.GetFileNameWithoutExtension(indexFile);
                Scan(indexFile, theme, "../", map);
            }
        }

        var memoryIndex = Path.Combine(root, "MEMORY.md");
        if (File.Exists(memoryIndex))
            Scan(memoryIndex, "MEMORY", "", map);

        return map;
    }

    private static readonly Regex LinkRe = new(
        @"\[([^\]]*)\]\((?<prefix>\.\./)?(?<file>[A-Za-z0-9_.\-]+\.md)\)",
        RegexOptions.Compiled
    );

    private static void Scan(
        string indexPath,
        string theme,
        string expectedPrefix,
        Dictionary<string, Placement> map
    )
    {
        foreach (var line in ReadLines(indexPath))
        {
            foreach (Match m in LinkRe.Matches(line))
            {
                var prefix = m.Groups["prefix"].Success ? "../" : "";
                if (prefix != expectedPrefix)
                    continue;

                var file = m.Groups["file"].Value;
                if (map.ContainsKey(file))
                    continue;

                var title = m.Groups[1].Value.Trim();
                var hook = line[(m.Index + m.Length)..].Trim();
                hook = StripLeadingDash(hook);

                map[file] = new Placement(theme, title, hook);
            }
        }
    }

    private static string StripLeadingDash(string s)
    {
        s = s.TrimStart();
        foreach (var dash in new[] { "\u2014", "\u2013", "-" })
        {
            if (s.StartsWith(dash, StringComparison.Ordinal))
            {
                s = s[dash.Length..].TrimStart();
                break;
            }
        }
        return s.Trim();
    }

    private static readonly Regex TypeRe = new(@"^\s+type:\s*(?<v>.+?)\s*$", RegexOptions.Compiled);

    private static (string Name, string Description, string Type) ReadFrontMatter(string path)
    {
        var name = "";
        var description = "";
        var type = "";

        var lines = ReadLines(path);
        if (lines.Count == 0 || lines[0].Trim() != "---")
            return (name, description, type);

        for (var i = 1; i < lines.Count; i++)
        {
            var line = lines[i];
            if (line.Trim() == "---")
                break;

            if (line.StartsWith("name:", StringComparison.Ordinal))
            {
                name = Unquote(line["name:".Length..]);
            }
            else if (line.StartsWith("description:", StringComparison.Ordinal))
            {
                description = Unquote(line["description:".Length..]);
            }
            else
            {
                var m = TypeRe.Match(line);
                if (m.Success && type.Length == 0)
                    type = Unquote(m.Groups["v"].Value);
            }
        }

        return (name, description, type);
    }

    private static string Unquote(string s)
    {
        s = s.Trim();
        if (s.Length >= 2 && s[0] == '"' && s[^1] == '"')
            s = s[1..^1];
        return s.Trim();
    }

    private static List<string> ReadLines(string path)
    {
        var text = File.ReadAllText(path);
        return text.Replace("\r\n", "\n").Replace('\r', '\n').Split('\n').ToList();
    }
}
