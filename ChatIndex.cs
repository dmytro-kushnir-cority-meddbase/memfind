using System.Buffers;
using System.Diagnostics;
using System.IO.MemoryMappedFiles;
using System.Numerics.Tensors;
using System.Runtime.InteropServices;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace MemFind;

internal sealed class ChatFileState
{
    [JsonPropertyName("path")] public string Path { get; set; } = "";
    [JsonPropertyName("length")] public long Length { get; set; }
    [JsonPropertyName("lastWriteUtc")] public string LastWriteUtc { get; set; } = "";
    [JsonPropertyName("itemsFrom")] public long ItemsFrom { get; set; }
    [JsonPropertyName("itemsCount")] public int ItemsCount { get; set; }
}

internal sealed class ChatState
{
    [JsonPropertyName("model")] public string Model { get; set; } = "";
    [JsonPropertyName("dim")] public int Dim { get; set; } = Embedder.Dim;
    [JsonPropertyName("includeAgents")] public bool IncludeAgents { get; set; }
    [JsonPropertyName("files")] public List<ChatFileState> Files { get; set; } = [];
}

internal static class ChatIndex
{
    private const int VectorBytes = Embedder.Dim * sizeof(float);

    public static string Dir(string root) => System.IO.Path.Combine(root, ".memfind");
    public static string VecPath(string root) => System.IO.Path.Combine(Dir(root), "chats.vec");
    public static string MetaPath(string root) => System.IO.Path.Combine(Dir(root), "chats.meta.jsonl");
    public static string StatePath(string root) => System.IO.Path.Combine(Dir(root), "chats.state.json");

    public static ChatState LoadState(string root)
    {
        var path = StatePath(root);
        if (!File.Exists(path)) return new ChatState();
        return JsonSerializer.Deserialize(File.ReadAllText(path), Ctx.Default.ChatState) ?? new ChatState();
    }

    public static int Build(string root, bool rebuild, bool includeAgents, bool int8)
    {
        var projects = Chats.ProjectsRoot;
        if (!Directory.Exists(projects))
        {
            Console.Error.WriteLine($"transcripts not found: {projects}");
            return 1;
        }

        var dir = Dir(root);
        Directory.CreateDirectory(dir);
        File.WriteAllText(System.IO.Path.Combine(dir, ".gitignore"), "*\n");

        var model = int8 ? Embedder.Int8ModelName : Embedder.Fp32ModelName;
        var state = rebuild ? new ChatState() : LoadState(root);

        if (state.Files.Count > 0 && (state.Model != model || state.IncludeAgents != includeAgents))
        {
            Console.Error.WriteLine($"chats index was built with model={state.Model} includeAgents={state.IncludeAgents} — rebuilding");
            rebuild = true;
            state = new ChatState();
        }

        state.Model = model;
        state.IncludeAgents = includeAgents;

        if (rebuild)
        {
            state.Files.Clear();
            if (File.Exists(VecPath(root))) File.Delete(VecPath(root));
            if (File.Exists(MetaPath(root))) File.Delete(MetaPath(root));
        }

        var total = Stopwatch.StartNew();
        var files = Chats.Discover(projects, includeAgents);

        var known = new Dictionary<string, List<ChatFileState>>(StringComparer.OrdinalIgnoreCase);
        foreach (var record in state.Files)
        {
            if (!known.TryGetValue(record.Path, out var list)) known[record.Path] = list = [];
            list.Add(record);
        }

        var work = new List<(string Path, string Relative, long From, long Length, string LastWrite)>();
        var present = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        var dropped = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        var skipped = 0;

        foreach (var path in files)
        {
            var relative = System.IO.Path.GetRelativePath(projects, path);
            present.Add(relative);

            var info = new FileInfo(path);
            var lastWrite = info.LastWriteTimeUtc.ToString("o");

            if (known.TryGetValue(relative, out var records))
            {
                var last = records.MaxBy(r => r.Length)!;
                if (info.Length == last.Length && lastWrite == last.LastWriteUtc) { skipped++; continue; }
                if (info.Length > last.Length)
                {
                    work.Add((path, relative, last.Length, info.Length, lastWrite));
                    continue;
                }
                dropped.Add(relative);
            }

            work.Add((path, relative, 0L, info.Length, lastWrite));
        }

        foreach (var relative in known.Keys)
            if (!present.Contains(relative))
                dropped.Add(relative);

        if (dropped.Count > 0)
        {
            Console.Error.WriteLine($"re-indexing {dropped.Count} changed/removed transcript(s)");
            Compact(root, state, dropped);
        }

        var existing = File.Exists(VecPath(root)) ? new FileInfo(VecPath(root)).Length / VectorBytes : 0;

        using var embedder = Embedder.Load(int8);

        var parse = Stopwatch.StartNew();
        var results = new ParseResult[work.Count];

        Parallel.For(0, work.Count, new ParallelOptions { MaxDegreeOfParallelism = Embedder.Threads }, i =>
        {
            var item = work[i];
            results[i] = Chats.ParseFile(item.Path, item.Relative, item.From, embedder, includeAgents);
        });

        parse.Stop();

        long lines = 0;
        var items = new List<ChatItem>();
        var next = existing;

        for (var i = 0; i < work.Count; i++)
        {
            var result = results[i];
            lines += result.Lines;
            if (result.Items.Count > 0)
            {
                state.Files.Add(new ChatFileState
                {
                    Path = work[i].Relative,
                    Length = result.Consumed,
                    LastWriteUtc = work[i].LastWrite,
                    ItemsFrom = next,
                    ItemsCount = result.Items.Count,
                });
                next += result.Items.Count;
                items.AddRange(result.Items);
            }
            else
            {
                state.Files.Add(new ChatFileState
                {
                    Path = work[i].Relative,
                    Length = result.Consumed,
                    LastWriteUtc = work[i].LastWrite,
                    ItemsFrom = next,
                    ItemsCount = 0,
                });
            }
        }

        var users = items.Count(i => i.Role == "user");
        var assistants = items.Count - users;

        Console.Error.WriteLine(
            $"read {work.Count} file(s) ({skipped} unchanged), {lines} lines, kept {items.Count} items " +
            $"(user {users}, assistant {assistants}) — parse+tokenize {parse.Elapsed.TotalSeconds:F1} s");

        if (items.Count > 0)
        {
            var tokens = new int[items.Count][];
            for (var i = 0; i < items.Count; i++) tokens[i] = items[i].Tokens;

            var vectors = new float[(long)items.Count * Embedder.Dim];
            var embed = Stopwatch.StartNew();
            var reported = 0;
            embedder.EmbedMany(tokens, vectors, done =>
            {
                if (done - reported < 20000) return;
                reported = done;
                Console.Error.WriteLine($"  embedded {done} / {items.Count} in {embed.Elapsed.TotalSeconds:F0} s");
            });
            embed.Stop();

            Console.Error.WriteLine(
                $"embedded {items.Count} items in {embed.Elapsed.TotalSeconds:F1} s " +
                $"({items.Count / Math.Max(embed.Elapsed.TotalSeconds, 0.001):F0} items/s, " +
                $"1 session x {embedder.IntraOpThreads} intra-op threads)");

            AppendVectors(VecPath(root), vectors);
            AppendMeta(MetaPath(root), items, existing);
        }

        File.WriteAllText(StatePath(root), JsonSerializer.Serialize(state, Ctx.Default.ChatState));

        total.Stop();
        Console.Error.WriteLine(
            $"chats: {next} items, {state.Files.Count} file record(s), " +
            $"{Size(VecPath(root)) / 1048576} MB vectors, " +
            $"{Size(MetaPath(root)) / 1048576} MB metadata " +
            $"in {total.Elapsed.TotalSeconds:F1} s");

        return 0;
    }

    public static long Size(string path) => File.Exists(path) ? new FileInfo(path).Length : 0;

    private static void AppendVectors(string path, float[] vectors)
    {
        using var stream = new FileStream(path, FileMode.Append, FileAccess.Write, FileShare.None, 1 << 20);

        const int slice = 1 << 22;
        for (var from = 0; from < vectors.Length; from += slice)
        {
            var count = Math.Min(slice, vectors.Length - from);
            stream.Write(MemoryMarshal.AsBytes(vectors.AsSpan(from, count)));
        }
    }

    private static void AppendMeta(string path, List<ChatItem> items, long from)
    {
        using var stream = new FileStream(path, FileMode.Append, FileAccess.Write, FileShare.None, 1 << 20);
        var buffer = new ArrayBufferWriter<byte>(4096);
        using var writer = new Utf8JsonWriter(buffer, new JsonWriterOptions { SkipValidation = true });

        for (var i = 0; i < items.Count; i++)
        {
            WriteMeta(writer, from + i, items[i]);
            stream.Write(buffer.WrittenSpan);
            stream.WriteByte((byte)'\n');
            buffer.Clear();
            writer.Reset(buffer);
        }
    }

    private static void WriteMeta(Utf8JsonWriter writer, long index, ChatItem item)
    {
        writer.WriteStartObject();
        writer.WriteNumber("i"u8, index);
        writer.WriteString("file"u8, item.File);
        writer.WriteString("session"u8, item.Session);
        writer.WriteString("cwd"u8, item.Cwd);
        writer.WriteString("role"u8, item.Role);
        writer.WriteString("ts"u8, item.Ts);
        writer.WriteString("text"u8, item.Preview);
        writer.WriteEndObject();
        writer.Flush();
    }

    private static void Compact(string root, ChatState state, HashSet<string> dropped)
    {
        var survivors = state.Files
            .Where(f => !dropped.Contains(f.Path))
            .OrderBy(f => f.ItemsFrom)
            .ToList();

        state.Files = survivors;

        var vec = VecPath(root);
        var meta = MetaPath(root);
        if (!File.Exists(vec)) return;

        var keep = survivors.Where(f => f.ItemsCount > 0).ToList();

        var vecTmp = vec + ".tmp";
        using (var source = new FileStream(vec, FileMode.Open, FileAccess.Read, FileShare.Read, 1 << 20))
        using (var target = new FileStream(vecTmp, FileMode.Create, FileAccess.Write, FileShare.None, 1 << 20))
        {
            var buffer = new byte[1 << 20];
            foreach (var record in keep)
            {
                source.Position = record.ItemsFrom * VectorBytes;
                var remaining = (long)record.ItemsCount * VectorBytes;
                while (remaining > 0)
                {
                    var n = source.Read(buffer, 0, (int)Math.Min(buffer.Length, remaining));
                    if (n <= 0) break;
                    target.Write(buffer, 0, n);
                    remaining -= n;
                }
            }
        }

        var ranges = keep
            .Select(r => (From: r.ItemsFrom, To: r.ItemsFrom + r.ItemsCount))
            .OrderBy(r => r.From)
            .ToList();

        var metaTmp = meta + ".tmp";
        if (File.Exists(meta))
        {
            using var source = new FileStream(meta, FileMode.Open, FileAccess.Read, FileShare.Read, 1 << 20, FileOptions.SequentialScan);
            using var target = new FileStream(metaTmp, FileMode.Create, FileAccess.Write, FileShare.None, 1 << 20);
            using var reader = new StreamReader(source);

            long index = 0;
            long written = 0;
            long expected = ranges.Sum(r => r.To - r.From);
            var range = 0;

            string? line;
            while ((line = reader.ReadLine()) is not null)
            {
                while (range < ranges.Count && index >= ranges[range].To) range++;
                if (range >= ranges.Count) break;

                if (index >= ranges[range].From)
                {
                    var comma = line.IndexOf(',');
                    var rest = comma < 0 ? "}" : line[comma..];
                    target.Write(System.Text.Encoding.UTF8.GetBytes($"{{\"i\":{written}{rest}\n"));
                    written++;
                }

                index++;
            }

            if (written != expected)
                Console.Error.WriteLine($"warning: metadata had {written} of {expected} expected lines");
        }

        File.Move(vecTmp, vec, overwrite: true);
        if (File.Exists(metaTmp)) File.Move(metaTmp, meta, overwrite: true);

        long position = 0;
        foreach (var record in survivors)
        {
            record.ItemsFrom = position;
            position += record.ItemsCount;
        }
    }

    public static unsafe int Search(string root, string query, int k)
    {
        var vec = VecPath(root);
        if (!File.Exists(vec) || new FileInfo(vec).Length == 0)
        {
            Console.Error.WriteLine("no chat index — run: memfind index --chats-only");
            return 1;
        }

        var state = LoadState(root);
        var count = (int)(new FileInfo(vec).Length / VectorBytes);

        using var embedder = Embedder.Load(state.Model != Embedder.Fp32ModelName);
        var q = embedder.EmbedQuery(query);

        var heap = new TopK(k);

        using (var map = MemoryMappedFile.CreateFromFile(vec, FileMode.Open, null, 0, MemoryMappedFileAccess.Read))
        using (var view = map.CreateViewAccessor(0, 0, MemoryMappedFileAccess.Read))
        {
            byte* pointer = null;
            view.SafeMemoryMappedViewHandle.AcquirePointer(ref pointer);
            try
            {
                for (var i = 0; i < count; i++)
                {
                    var row = new ReadOnlySpan<float>(pointer + (long)i * VectorBytes, Embedder.Dim);
                    heap.Add(TensorPrimitives.Dot(q.AsSpan(), row), i);
                }
            }
            finally
            {
                view.SafeMemoryMappedViewHandle.ReleasePointer();
            }
        }

        var hits = heap.Sorted();
        var meta = ReadMeta(MetaPath(root), hits.Select(h => h.Index).ToHashSet());

        var rows = new List<(float Score, string Role, string Slug, string Session, string Time, string Text)>();
        foreach (var (score, index) in hits)
        {
            if (!meta.TryGetValue(index, out var line)) continue;

            using var document = JsonDocument.Parse(line);
            var element = document.RootElement;
            var file = Get(element, "file");
            var slug = file.Split('\\', '/')[0];
            var session = Get(element, "session");
            var ts = Get(element, "ts");
            var time = DateTimeOffset.TryParse(ts, out var parsed)
                ? parsed.ToLocalTime().ToString("yyyy-MM-dd HH:mm")
                : ts;

            rows.Add((score, Get(element, "role"), slug, session.Length >= 8 ? session[..8] : session, time, Get(element, "text")));
        }

        var roleWidth = rows.Count == 0 ? 0 : rows.Max(r => r.Role.Length);
        var slugWidth = rows.Count == 0 ? 0 : rows.Max(r => r.Slug.Length);

        foreach (var row in rows)
        {
            Console.WriteLine($"{row.Score:F3}  {row.Time}  {row.Role.PadRight(roleWidth)}   {row.Slug.PadRight(slugWidth)}   {row.Session}");
            Console.WriteLine($"       {Truncate(row.Text, 160)}");
        }

        return 0;
    }

    private static string Get(JsonElement element, string name) =>
        element.TryGetProperty(name, out var value) ? value.GetString() ?? "" : "";

    private static string Truncate(string s, int max)
    {
        s = s.Replace('\n', ' ').Replace('\r', ' ').Replace('\t', ' ').Trim();
        while (s.Contains("  ", StringComparison.Ordinal)) s = s.Replace("  ", " ", StringComparison.Ordinal);
        return s.Length <= max ? s : s[..max] + "...";
    }

    private static Dictionary<int, string> ReadMeta(string path, HashSet<int> wanted)
    {
        var found = new Dictionary<int, string>();
        if (!File.Exists(path) || wanted.Count == 0) return found;

        using var stream = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.ReadWrite, 1 << 20, FileOptions.SequentialScan);
        using var reader = new StreamReader(stream);

        var index = 0;
        string? line;
        while ((line = reader.ReadLine()) is not null)
        {
            if (wanted.Contains(index)) found[index] = line;
            index++;
            if (found.Count == wanted.Count) break;
        }

        return found;
    }

    private struct TopK(int k)
    {
        private readonly float[] _scores = new float[k];
        private readonly int[] _indices = new int[k];
        private int _count = 0;

        public void Add(float score, int index)
        {
            if (_count < _scores.Length)
            {
                _scores[_count] = score;
                _indices[_count] = index;
                _count++;
                if (_count == _scores.Length) Heapify();
                return;
            }

            if (score <= _scores[0]) return;
            _scores[0] = score;
            _indices[0] = index;
            SiftDown(0);
        }

        private void Heapify()
        {
            for (var i = _count / 2 - 1; i >= 0; i--) SiftDown(i);
        }

        private void SiftDown(int i)
        {
            while (true)
            {
                var smallest = i;
                var left = 2 * i + 1;
                var right = left + 1;

                if (left < _count && _scores[left] < _scores[smallest]) smallest = left;
                if (right < _count && _scores[right] < _scores[smallest]) smallest = right;
                if (smallest == i) return;

                (_scores[i], _scores[smallest]) = (_scores[smallest], _scores[i]);
                (_indices[i], _indices[smallest]) = (_indices[smallest], _indices[i]);
                i = smallest;
            }
        }

        public List<(float Score, int Index)> Sorted()
        {
            var hits = new List<(float, int)>(_count);
            for (var i = 0; i < _count; i++) hits.Add((_scores[i], _indices[i]));
            hits.Sort((a, b) => b.Item1.CompareTo(a.Item1));
            return hits;
        }
    }
}
