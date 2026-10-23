using System.Text.Json;

namespace MemFind;

/// <summary>Session metadata and the canonical message reader, shared by indexing and show.</summary>
internal sealed record Transcript(bool Codex, bool Agent, string Session, string Cwd)
{
    public static bool IsCompressed(string path) => path.EndsWith(".jsonl.zst", StringComparison.OrdinalIgnoreCase);

    public static Stream Open(string path)
    {
        var stream = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.ReadWrite, 1 << 20,
            FileOptions.SequentialScan);
        if (!IsCompressed(path))
        {
            return stream;
        }
        try
        {
            return new ZstdSharp.DecompressionStream(stream);
        }
        catch
        {
            stream.Dispose();
            throw;
        }
    }

    public string Source => Codex ? (Agent ? "codex-agent" : "codex") : (Agent ? "agent" : "chat");

    public static Transcript Inspect(string path)
    {
        var agent = path.Split(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar)
            .Any(part => part.Equals("subagents", StringComparison.OrdinalIgnoreCase));
        using var stream = Open(path);
        using var reader = new StreamReader(stream);
        // Rollouts start with session_meta. Bound the header read just like the message reader.
        var header = new char[65536];
        var length = 0;
        int next;
        while (length < header.Length && (next = reader.Read()) >= 0 && next != '\n')
        {
            header[length++] = (char)next;
        }
        try
        {
            using var document = JsonDocument.Parse(new string(header, 0, length));
            var root = document.RootElement;
            if (Get(root, "type") == "session_meta" && root.TryGetProperty("payload", out var payload)
                && payload.ValueKind == JsonValueKind.Object)
            {
                agent |= payload.TryGetProperty("source", out var source)
                    && ((source.ValueKind == JsonValueKind.Object && source.TryGetProperty("subagent", out _))
                        || (source.ValueKind == JsonValueKind.String && source.GetString() == "subagent"));
                return new Transcript(true, agent, Get(payload, "id"), Get(payload, "cwd"));
            }
        }
        catch (JsonException) { }
        return new Transcript(false, agent, "", "");
    }

    public bool TryRead(ReadOnlySpan<byte> line, long number, bool includeAgents, out Chats.Message message)
    {
        message = Chats.Message.Empty;
        if (Agent && !includeAgents)
        {
            return false;
        }
        if (!Codex)
        {
            return Chats.TryReadKept(line, includeAgents, out message);
        }
        try
        {
            var reader = new Utf8JsonReader(line);
            using var document = JsonDocument.ParseValue(ref reader);
            var root = document.RootElement;
            if (Get(root, "type") != "response_item" || !root.TryGetProperty("payload", out var payload)
                || Get(payload, "type") != "message")
            {
                return false;
            }
            var role = Get(payload, "role");
            if (role is not ("user" or "assistant") || !payload.TryGetProperty("content", out var content)
                || content.ValueKind != JsonValueKind.Array)
            {
                return false;
            }
            var texts = new List<string>();
            foreach (var part in content.EnumerateArray())
            {
                if (Get(part, "type") is "input_text" or "output_text" && Get(part, "text") is { Length: > 0 } text)
                {
                    texts.Add(text);
                }
            }
            var body = string.Join("\n\n", texts).Trim();
            if (body.Length == 0)
            {
                return false;
            }
            message = new Chats.Message
            {
                Role = role, Text = body, Ts = Get(root, "timestamp"), Session = Session, Cwd = Cwd,
                Uuid = Get(payload, "id") is { Length: > 0 } id ? id : $"{Session}:{number}", Model = "",
            };
            return true;
        }
        catch (JsonException)
        {
            return false;
        }
    }

    private static string Get(JsonElement element, string name) =>
        element.ValueKind == JsonValueKind.Object && element.TryGetProperty(name, out var value)
            && value.ValueKind == JsonValueKind.String ? value.GetString() ?? "" : "";
}
