namespace MemFind;

internal static class IndexLock
{
    // Filesystem aliases resolve to the same file, including case-insensitive volumes.
    internal static FileStream? TryAcquire(string store) =>
        FileLock.TryAcquire(Path.Combine(store, ".memfind", "index.lock"));
}
