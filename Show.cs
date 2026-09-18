namespace MemFind;

/// <summary>Prints the message a hit's <c>path:line</c> anchor points at, with its neighbours.</summary>
internal static class Show
{
    private sealed record Shown(
        string Role,
        string Ts,
        long Line,
        string Uuid,
        bool Anchor,
        string Text
    );

    public static int Run(string path, long anchor, int context, bool json, bool includeAgents)
    {
        if (!File.Exists(path))
        {
            Console.Error.WriteLine($"no such transcript: {path}");
            return 1;
        }

        var before = new Queue<Shown>();
        var shown = new List<Shown>();
        var found = false;
        var bad = false;

        Scan(path, includeAgents, anchor, context, before, shown, ref found, ref bad);

        if (bad)
        {
            Console.Error.WriteLine(
                $"line {anchor} of {path} is not an indexed user/assistant message"
            );
            return 1;
        }

        if (!found)
        {
            Console.Error.WriteLine($"no line {anchor} in {path}");
            return 1;
        }

        var messages = new List<Shown>(before);
        messages.AddRange(shown);

        if (json)
        {
            using var output = new JsonLines();
            foreach (var message in messages)
            {
                var writer = output.Writer;
                writer.WriteStartObject();
                writer.WriteString("role"u8, message.Role);
                writer.WriteString("ts"u8, message.Ts);
                writer.WriteNumber("line"u8, message.Line);
                writer.WriteString("uuid"u8, message.Uuid);
                writer.WriteBoolean("anchor"u8, message.Anchor);
                writer.WriteString("text"u8, message.Text);
                writer.WriteEndObject();
                output.Flush();
            }

            return 0;
        }

        for (var i = 0; i < messages.Count; i++)
        {
            if (i > 0)
                Console.WriteLine();
            var message = messages[i];
            var time = DateTimeOffset.TryParse(message.Ts, out var parsed)
                ? parsed.ToLocalTime().ToString("yyyy-MM-dd HH:mm")
                : message.Ts;
            Console.WriteLine(
                $"{(message.Anchor ? ">>>" : "---")} {message.Role}  {time}  line {message.Line}"
            );
            Console.WriteLine(message.Text);
        }

        return 0;
    }

    private static void Scan(
        string path,
        bool includeAgents,
        long anchor,
        int context,
        Queue<Shown> before,
        List<Shown> shown,
        ref bool found,
        ref bool bad
    )
    {
        using var stream = new FileStream(
            path,
            FileMode.Open,
            FileAccess.Read,
            FileShare.ReadWrite,
            1 << 20,
            FileOptions.SequentialScan
        );

        var buffer = new byte[Chats.ReadChunk];
        var filled = 0;
        long number = 0;
        var done = false;

        while (!done)
        {
            if (filled == buffer.Length)
            {
                if (buffer.Length < Chats.MaxLineBytes)
                {
                    Array.Resize(ref buffer, buffer.Length * 2);
                }
                else
                {
                    // Mirrors the indexer: an over-long line is skipped but still counted.
                    number++;
                    if (number == anchor)
                    {
                        bad = true;
                        return;
                    }
                    filled = 0;

                    int b;
                    while ((b = stream.ReadByte()) >= 0)
                        if (b == '\n')
                            break;
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
                number++;

                if (line.Length > 0 && line[^1] == (byte)'\r')
                    line = line[..^1];

                var message = Chats.Message.Empty;
                var keep = line.Length > 0 && Chats.TryReadKept(line, includeAgents, out message);
                if (number == anchor && !keep)
                {
                    bad = true;
                    return;
                }

                if (keep)
                {
                    var item = new Shown(
                        message.Role,
                        message.Ts,
                        number,
                        message.Uuid,
                        number == anchor,
                        message.Text
                    );

                    if (number == anchor)
                    {
                        found = true;
                        shown.Add(item);
                    }
                    else if (found)
                    {
                        shown.Add(item);
                    }
                    else
                    {
                        before.Enqueue(item);
                        if (before.Count > context)
                            before.Dequeue();
                    }
                }

                if (found && shown.Count > context)
                {
                    done = true;
                    break;
                }
            }

            filled -= start;
            if (filled > 0)
                Buffer.BlockCopy(buffer, start, buffer, 0, filled);
        }
    }
}
