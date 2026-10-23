using System.Security.Cryptography;
using System.Text;

namespace MemFind;

/// <summary>Coalesces identical indexing requests; IndexLock serializes all writers to a store.</summary>
internal sealed class IndexQueue
{
    private readonly string _store;
    private readonly string _workerLock;
    internal string Requested { get; }
    internal string Log { get; }

    public IndexQueue(string store, IEnumerable<string> options)
    {
        _store = store;
        var key = Convert.ToHexStringLower(SHA256.HashData(Encoding.UTF8.GetBytes(string.Join('\0', options))))[..12];
        var prefix = Path.Combine(store, ".memfind", "index-" + key);
        Requested = prefix + ".requested";
        _workerLock = prefix + ".worker.lock";
        Log = prefix + ".log";
    }

    public void Request(Action start)
    {
        MarkRequested();
        // The worker, not this short-lived launcher, owns the lock for the full drain.
        using (var available = FileLock.TryAcquire(_workerLock))
        {
            if (available is null)
                return;
        }
        start();
    }

    public int Drain(Func<int> build)
    {
        while (true)
        {
            using (var worker = FileLock.TryAcquire(_workerLock))
            {
                if (worker is null)
                    return 0;
                while (File.Exists(Requested))
                {
                    using var writer = IndexLock.TryAcquire(_store);
                    if (writer is null)
                    {
                        // A foreground index or a different option set may already own the writer.
                        Thread.Sleep(1000);
                        continue;
                    }
                    // Events before this deletion are covered by the upcoming scan; later ones
                    // create a new marker and cause one more incremental pass.
                    File.Delete(Requested);
                    try
                    {
                        var code = build();
                        if (code == 0)
                            continue;
                        MarkRequested();
                        return code;
                    }
                    catch
                    {
                        MarkRequested();
                        throw;
                    }
                }
            }
            // Close the lost-wakeup window: a requester could have seen the worker still locked
            // after its last empty check. After unlock, either we see its marker or it starts a worker.
            if (!File.Exists(Requested))
                return 0;
        }
    }

    private void MarkRequested()
    {
        Directory.CreateDirectory(Path.GetDirectoryName(Requested)!);
        using var marker = new FileStream(Requested, FileMode.OpenOrCreate, FileAccess.Write,
            FileShare.ReadWrite | FileShare.Delete);
    }
}
