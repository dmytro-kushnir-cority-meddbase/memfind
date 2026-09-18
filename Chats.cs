using System.Text.Json;

namespace MemFind;

internal sealed class ChatItem
{
    public string File = "";
    public string Session = "";
    public string Cwd = "";
    public string Role = "";
    public string Ts = "";
    public string Uuid = "";
    public long Line;
    public int Chunk;
    public string Preview = "";
    public int[] Tokens = [];
}

internal sealed class ParseResult
{
    public List<ChatItem> Items = [];
    public long Consumed;
    public long Lines;
}

internal static class Chats
{
    public const int MinChars = 40;
    public const int AssistantMinChars = 200;
    private const int TargetChars = 1200;
    private const int MaxChars = 1600;
    private const int PreviewChars = 300;

    /// <summary>Read window; transcripts reach tens of MB, so they are never held whole.</summary>
    public const int ReadChunk = 1 << 22;

    /// <summary>A single line longer than this is skipped rather than buffered.</summary>
    public const int MaxLineBytes = 1 << 26;

    public static string ProjectsRoot =>
        Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
            ".claude",
            "projects"
        );

    public static List<string> Discover(string root, bool includeAgents)
    {
        var files = new List<string>();
        if (Directory.Exists(root))
            Walk(root, includeAgents, files);
        files.Sort(StringComparer.OrdinalIgnoreCase);
        return files;
    }

    private static void Walk(string dir, bool includeAgents, List<string> files)
    {
        foreach (var file in Directory.EnumerateFiles(dir, "*.jsonl"))
            files.Add(file);

        foreach (var sub in new DirectoryInfo(dir).EnumerateDirectories())
        {
            var name = sub.Name;
            if (name.StartsWith('_') || name.StartsWith('.'))
                continue;
            if (!includeAgents && name.Equals("subagents", StringComparison.OrdinalIgnoreCase))
                continue;

            // Each project dir holds a `memory` junction into the memory store, where this index writes
            // chats.meta.jsonl — following it would index our own output, once per junction.
            if (sub.Attributes.HasFlag(FileAttributes.ReparsePoint))
                continue;

            Walk(sub.FullName, includeAgents, files);
        }
    }

    public static ParseResult ParseFile(
        string path,
        string relative,
        long from,
        long fromLine,
        Embedder embedder,
        bool includeAgents
    )
    {
        var result = new ParseResult { Consumed = from };

        using var stream = new FileStream(
            path,
            FileMode.Open,
            FileAccess.Read,
            FileShare.ReadWrite,
            1 << 20,
            FileOptions.SequentialScan
        );
        var length = stream.Length;
        if (from >= length)
        {
            result.Consumed = length;
            return result;
        }

        stream.Position = from;

        var buffer = new byte[ReadChunk];
        var filled = 0;
        var consumed = from;

        while (true)
        {
            if (filled == buffer.Length)
            {
                if (buffer.Length < MaxLineBytes)
                {
                    Array.Resize(ref buffer, buffer.Length * 2);
                }
                else
                {
                    result.Lines++;
                    consumed += filled;
                    filled = 0;

                    int b;
                    while ((b = stream.ReadByte()) >= 0)
                    {
                        consumed++;
                        if (b == '\n')
                            break;
                    }
                    if (b < 0)
                        break;
                    continue;
                }
            }

            var read = stream.Read(buffer, filled, buffer.Length - filled);
            if (read <= 0)
                break;
            filled += read;

            var start = 0;
            while (true)
            {
                var nl = buffer.AsSpan(start, filled - start).IndexOf((byte)'\n');
                if (nl < 0)
                    break;

                var line = buffer.AsSpan(start, nl);
                start += nl + 1;
                result.Lines++;

                if (line.Length > 0 && line[^1] == (byte)'\r')
                    line = line[..^1];
                if (line.Length == 0)
                    continue;

                ReadLine(line, relative, fromLine + result.Lines, includeAgents, embedder, result);
            }

            consumed += start;
            filled -= start;
            if (filled > 0)
                Buffer.BlockCopy(buffer, start, buffer, 0, filled);
        }

        result.Consumed = consumed;
        return result;
    }

    private static void ReadLine(
        ReadOnlySpan<byte> line,
        string relative,
        long number,
        bool includeAgents,
        Embedder embedder,
        ParseResult result
    )
    {
        if (!TryReadKept(line, includeAgents, out var message))
            return;

        var chunks = Chunk(message.Text);
        for (var i = 0; i < chunks.Count; i++)
        {
            var chunk = chunks[i];
            result.Items.Add(
                new ChatItem
                {
                    File = relative,
                    Session = message.Session,
                    Cwd = message.Cwd,
                    Role = message.Role,
                    Ts = message.Ts,
                    Uuid = message.Uuid,
                    Line = number,
                    Chunk = i,
                    Preview = chunk.Length <= PreviewChars ? chunk : chunk[..PreviewChars],
                    Tokens = embedder.Encode(chunk, Embedder.ChatMaxTokens),
                }
            );
        }
    }

    /// <summary>The indexer's keep/skip filter, shared with <c>memfind show</c>.</summary>
    public static bool TryReadKept(ReadOnlySpan<byte> line, bool includeAgents, out Message message)
    {
        try
        {
            if (!IsUserOrAssistant(line))
            {
                message = Message.Empty;
                return false;
            }
            return TryRead(line, includeAgents, out message);
        }
        catch (JsonException)
        {
            message = Message.Empty;
            return false;
        }
    }

    private static bool IsUserOrAssistant(ReadOnlySpan<byte> line)
    {
        var reader = new Utf8JsonReader(line, isFinalBlock: true, state: default);
        if (!reader.Read() || reader.TokenType != JsonTokenType.StartObject)
            return false;

        while (reader.Read() && reader.TokenType != JsonTokenType.EndObject)
        {
            if (reader.TokenType != JsonTokenType.PropertyName)
                return false;

            var isType = reader.ValueTextEquals("type"u8);
            if (!reader.Read())
                return false;

            if (isType)
                return reader.TokenType == JsonTokenType.String
                    && (reader.ValueTextEquals("user"u8) || reader.ValueTextEquals("assistant"u8));

            reader.Skip();
        }

        return false;
    }

    internal struct Message
    {
        public string Role;
        public string Model;
        public string Ts;
        public string Cwd;
        public string Session;
        public string Uuid;
        public string Text;

        public static Message Empty =>
            new()
            {
                Role = "",
                Model = "",
                Ts = "",
                Cwd = "",
                Session = "",
                Uuid = "",
                Text = "",
            };
    }

    private static bool TryRead(ReadOnlySpan<byte> line, bool includeAgents, out Message message)
    {
        message = Message.Empty;

        var reader = new Utf8JsonReader(line, isFinalBlock: true, state: default);
        if (!reader.Read() || reader.TokenType != JsonTokenType.StartObject)
            return false;

        List<string>? texts = null;

        while (reader.Read() && reader.TokenType != JsonTokenType.EndObject)
        {
            if (reader.TokenType != JsonTokenType.PropertyName)
                return false;

            if (reader.ValueTextEquals("isSidechain"u8))
            {
                if (!reader.Read())
                    return false;
                if (!includeAgents && reader.TokenType == JsonTokenType.True)
                    return false;
            }
            else if (reader.ValueTextEquals("isMeta"u8))
            {
                if (!reader.Read())
                    return false;
                if (reader.TokenType == JsonTokenType.True)
                    return false;
            }
            else if (reader.ValueTextEquals("timestamp"u8))
            {
                if (!reader.Read())
                    return false;
                if (reader.TokenType == JsonTokenType.String)
                    message.Ts = reader.GetString() ?? "";
                else
                    reader.Skip();
            }
            else if (reader.ValueTextEquals("cwd"u8))
            {
                if (!reader.Read())
                    return false;
                if (reader.TokenType == JsonTokenType.String)
                    message.Cwd = reader.GetString() ?? "";
                else
                    reader.Skip();
            }
            else if (reader.ValueTextEquals("uuid"u8))
            {
                if (!reader.Read())
                    return false;
                if (reader.TokenType == JsonTokenType.String)
                    message.Uuid = reader.GetString() ?? "";
                else
                    reader.Skip();
            }
            else if (reader.ValueTextEquals("sessionId"u8))
            {
                if (!reader.Read())
                    return false;
                if (reader.TokenType == JsonTokenType.String)
                    message.Session = reader.GetString() ?? "";
                else
                    reader.Skip();
            }
            else if (reader.ValueTextEquals("message"u8))
            {
                if (!reader.Read())
                    return false;
                if (reader.TokenType == JsonTokenType.StartObject)
                    ReadMessageObject(ref reader, ref message, ref texts);
                else
                    reader.Skip();
            }
            else
            {
                if (!reader.Read())
                    return false;
                reader.Skip();
            }
        }

        if (texts is null || texts.Count == 0)
            return false;
        if (message.Model == "<synthetic>")
            return false;
        if (message.Role.Length == 0)
            message.Role = "user";

        var text = Clean(string.Join("\n\n", texts));
        if (text.Length < (message.Role == "assistant" ? AssistantMinChars : MinChars))
            return false;
        if (text == "No response requested.")
            return false;
        if (text.StartsWith("<command-name>", StringComparison.Ordinal))
            return false;
        if (text.StartsWith("<local-command-stdout>", StringComparison.Ordinal))
            return false;
        if (text.StartsWith("<local-command-caveat>", StringComparison.Ordinal))
            return false;
        if (text.StartsWith("[Request interrupted", StringComparison.Ordinal))
            return false;
        if (text.StartsWith("<task-notification>", StringComparison.Ordinal))
            return false;
        if (text.StartsWith("[SYSTEM NOTIFICATION", StringComparison.Ordinal))
            return false;

        message.Text = text;
        return true;
    }

    private static void ReadMessageObject(
        ref Utf8JsonReader reader,
        ref Message message,
        ref List<string>? texts
    )
    {
        while (reader.Read() && reader.TokenType != JsonTokenType.EndObject)
        {
            if (reader.TokenType != JsonTokenType.PropertyName)
                return;

            if (reader.ValueTextEquals("role"u8))
            {
                if (!reader.Read())
                    return;
                if (reader.TokenType == JsonTokenType.String)
                    message.Role = reader.GetString() ?? "";
                else
                    reader.Skip();
            }
            else if (reader.ValueTextEquals("model"u8))
            {
                if (!reader.Read())
                    return;
                if (reader.TokenType == JsonTokenType.String)
                    message.Model = reader.GetString() ?? "";
                else
                    reader.Skip();
            }
            else if (reader.ValueTextEquals("content"u8))
            {
                if (!reader.Read())
                    return;
                ReadContent(ref reader, ref texts);
            }
            else
            {
                if (!reader.Read())
                    return;
                reader.Skip();
            }
        }
    }

    private static void ReadContent(ref Utf8JsonReader reader, ref List<string>? texts)
    {
        if (reader.TokenType == JsonTokenType.String)
        {
            (texts ??= []).Add(reader.GetString() ?? "");
            return;
        }

        if (reader.TokenType != JsonTokenType.StartArray)
        {
            reader.Skip();
            return;
        }

        while (reader.Read() && reader.TokenType != JsonTokenType.EndArray)
        {
            if (reader.TokenType != JsonTokenType.StartObject)
            {
                reader.Skip();
                continue;
            }

            var isText = false;
            string? text = null;

            while (reader.Read() && reader.TokenType != JsonTokenType.EndObject)
            {
                if (reader.TokenType != JsonTokenType.PropertyName)
                    return;

                var typeProperty = reader.ValueTextEquals("type"u8);
                var textProperty = reader.ValueTextEquals("text"u8);
                if (!reader.Read())
                    return;

                if (typeProperty)
                    isText =
                        reader.TokenType == JsonTokenType.String
                        && reader.ValueTextEquals("text"u8);
                else if (textProperty && reader.TokenType == JsonTokenType.String)
                    text = reader.GetString();
                else
                    reader.Skip();
            }

            if (isText && text is not null)
                (texts ??= []).Add(text);
        }
    }

    private static string Clean(string text)
    {
        while (true)
        {
            var start = text.IndexOf("<system-reminder>", StringComparison.Ordinal);
            if (start < 0)
                break;

            var end = text.IndexOf("</system-reminder>", start, StringComparison.Ordinal);
            text =
                end < 0
                    ? text[..start]
                    : text[..start] + text[(end + "</system-reminder>".Length)..];
        }

        return text.Trim();
    }

    public static List<string> Chunk(string text)
    {
        var chunks = new List<string>();
        var pos = 0;

        while (pos < text.Length)
        {
            if (text.Length - pos <= MaxChars)
            {
                var tail = text[pos..].Trim();
                if (tail.Length >= MinChars)
                    chunks.Add(tail);
                break;
            }

            var end = FindBreak(text, pos);
            var chunk = text[pos..end].Trim();
            if (chunk.Length >= MinChars)
                chunks.Add(chunk);
            pos = end;
        }

        return chunks;
    }

    private static int FindBreak(string text, int pos)
    {
        var low = pos + TargetChars / 2;
        var high = Math.Min(text.Length, pos + MaxChars - 200);

        var paragraph = text.LastIndexOf("\n\n", high - 1, high - low, StringComparison.Ordinal);
        if (paragraph >= low)
            return paragraph + 2;

        for (var i = high - 1; i > low; i--)
        {
            var c = text[i];
            if (c is not ('.' or '!' or '?' or ';' or '\n'))
                continue;
            if (i + 1 < text.Length && !char.IsWhiteSpace(text[i + 1]))
                continue;
            return i + 1;
        }

        return pos + TargetChars;
    }
}
