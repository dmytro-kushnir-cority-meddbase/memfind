namespace MemFind;

internal static class Paths
{
    private static string Home => Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);

    public static string CodexHome => Expand(Environment.GetEnvironmentVariable("CODEX_HOME")
        ?? Path.Combine(Home, ".codex"));

    public static string Store => Expand(Environment.GetEnvironmentVariable("MEMFIND_STORE")
        ?? Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "memfind", "memory"));

    public static string ClaudeProjects { get; set; } = Path.Combine(Home, ".claude", "projects");
    public static string CodexSessions { get; set; } = Path.Combine(CodexHome, "sessions");
    public static string CodexMemories { get; set; } = Path.Combine(CodexHome, "memories");

    public static StringComparer Comparer => OperatingSystem.IsWindows()
        ? StringComparer.OrdinalIgnoreCase : StringComparer.Ordinal;

    public static string Expand(string path)
    {
        if (path == "~")
        {
            path = Home;
        }
        else if (path.StartsWith("~/", StringComparison.Ordinal) || path.StartsWith("~\\", StringComparison.Ordinal))
        {
            path = Path.Combine(Home, path[2..]);
        }
        return Path.GetFullPath(Environment.ExpandEnvironmentVariables(path));
    }
}
