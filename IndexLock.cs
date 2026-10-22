namespace MemFind;

internal static class IndexLock
{
    // Keep the file in place: deleting it could let another process lock a new inode.
    // Filesystem aliases resolve to the same file, including case-insensitive volumes.
    internal static FileStream? TryAcquire(string store)
    {
        var dir = Path.Combine(store, ".memfind");
        Directory.CreateDirectory(dir);
        try
        {
            return new FileStream(Path.Combine(dir, "index.lock"), FileMode.OpenOrCreate,
                FileAccess.ReadWrite, FileShare.None);
        }
        // Windows uses sharing/lock violations; Unix preserves EWOULDBLOCK errno.
        catch (IOException e) when (OperatingSystem.IsWindows()
            ? (e.HResult & 0xffff) is 32 or 33
            : e.HResult == (OperatingSystem.IsMacOS() ? 35 : 11))
        {
            return null;
        }
    }
}
