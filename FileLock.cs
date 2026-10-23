namespace MemFind;

internal static class FileLock
{
    // Keep the file in place: deleting it could let another process lock a new inode.
    // File sharing locks work across processes in both CoreCLR and NativeAOT.
    internal static FileStream? TryAcquire(string path)
    {
        Directory.CreateDirectory(Path.GetDirectoryName(path)!);
        try
        {
            return new FileStream(path, FileMode.OpenOrCreate, FileAccess.ReadWrite, FileShare.None);
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
