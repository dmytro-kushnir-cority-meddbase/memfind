using System.Diagnostics;
using System.Text.Json;

namespace MemFind;

internal static class IndexHook
{
    public static int Run(string store, string[] options, Func<int> build, bool worker)
    {
        var queue = new IndexQueue(store, options);
        if (!worker)
        {
            try
            {
                using var input = JsonDocument.Parse(Console.In.ReadToEnd());
                if (!input.RootElement.TryGetProperty("hook_event_name", out var name)
                    || name.GetString() is not ("SessionStart" or "PreCompact"))
                    return 0;
                queue.Request(() =>
                {
                    if (BackgroundProcess.Start(["index-worker", .. options]) is null)
                        throw new IOException("Could not start the background indexer");
                });
            }
            catch (Exception e)
            {
                // Optional maintenance must never stop session startup or compaction.
                try { File.AppendAllText(queue.Log, $"{DateTimeOffset.UtcNow:O} request failed: {e.Message}\n"); }
                catch (Exception) { }
            }
            return 0;
        }

        Directory.CreateDirectory(Path.GetDirectoryName(queue.Log)!);
        using var log = new StreamWriter(new FileStream(queue.Log, FileMode.Append, FileAccess.Write,
            FileShare.ReadWrite)) { AutoFlush = true };
        var stdout = Console.Out;
        var stderr = Console.Error;
        Console.SetOut(log);
        Console.SetError(log);
        try
        {
            log.WriteLine($"{DateTimeOffset.UtcNow:O} worker pid={Environment.ProcessId} started");
            var code = queue.Drain(() =>
            {
                log.WriteLine($"{DateTimeOffset.UtcNow:O} incremental pass started");
                return build();
            });
            log.WriteLine($"{DateTimeOffset.UtcNow:O} worker finished, exit={code}");
            return code;
        }
        catch (Exception e)
        {
            log.WriteLine($"{DateTimeOffset.UtcNow:O} worker failed: {e}");
            return 1;
        }
        finally
        {
            Console.SetOut(stdout);
            Console.SetError(stderr);
        }
    }
}
