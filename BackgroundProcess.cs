using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Text;

namespace MemFind;

/// <summary>Starts this executable with independent standard streams and no caller working directory.</summary>
internal static partial class BackgroundProcess
{
    private const uint DetachedProcess = 0x8;
    private const uint CreateNewProcessGroup = 0x200;

    [StructLayout(LayoutKind.Sequential)]
    private struct StartupInfo
    {
        public int Size;
        public nint Reserved, Desktop, Title;
        public int X, Y, XSize, YSize, XCountChars, YCountChars, FillAttribute, Flags;
        public short ShowWindow, Reserved2Size;
        public nint Reserved2, StdInput, StdOutput, StdError;
    }

    [StructLayout(LayoutKind.Sequential)]
    private struct ProcessInformation
    {
        public nint Process, Thread;
        public int ProcessId, ThreadId;
    }

    [LibraryImport("kernel32.dll", EntryPoint = "CreateProcessW", StringMarshalling = StringMarshalling.Utf16)]
    [return: MarshalAs(UnmanagedType.Bool)]
    private static partial bool CreateProcess(string? application, char[] commandLine, nint processAttributes,
        nint threadAttributes, [MarshalAs(UnmanagedType.Bool)] bool inheritHandles, uint flags, nint environment,
        string? directory, in StartupInfo startup, out ProcessInformation information);

    [LibraryImport("kernel32.dll")]
    [return: MarshalAs(UnmanagedType.Bool)]
    private static partial bool CloseHandle(nint handle);

    public static int? Start(IEnumerable<string> arguments)
    {
        var program = Environment.ProcessPath!;
        var args = new List<string>();
        if (Path.GetFileNameWithoutExtension(program).Equals("dotnet", StringComparison.OrdinalIgnoreCase))
            args.Add(Path.GetFullPath(Environment.GetCommandLineArgs()[0]));
        args.AddRange(arguments);
        int pid;
        if (OperatingSystem.IsWindows())
        {
            // A resident must never inherit the hook's stdout: the caller waits for EOF on it.
            var startup = new StartupInfo { Size = Marshal.SizeOf<StartupInfo>() };
            var command = string.Join(" ", new[] { program }.Concat(args).Select(QuoteWindowsArgument)) + '\0';
            if (!CreateProcess(null, command.ToCharArray(), 0, 0, false,
                    DetachedProcess | CreateNewProcessGroup, 0, AppContext.BaseDirectory, startup, out var created))
                return null;
            CloseHandle(created.Process);
            CloseHandle(created.Thread);
            pid = created.ProcessId;
        }
        else
        {
            // Pass paths as argv, never shell text. exec keeps the PID; /dev/null survives the hook
            // exiting, unlike redirected .NET pipes. Ignore terminal hangup and release the caller's cwd.
            var start = new ProcessStartInfo("/bin/sh")
            {
                UseShellExecute = false, WorkingDirectory = AppContext.BaseDirectory,
                RedirectStandardInput = true, RedirectStandardOutput = true, RedirectStandardError = true,
            };
            start.ArgumentList.Add("-c");
            start.ArgumentList.Add("trap '' HUP; exec \"$@\" </dev/null >/dev/null 2>&1");
            start.ArgumentList.Add("memfind-background");
            start.ArgumentList.Add(program);
            foreach (var arg in args)
                start.ArgumentList.Add(arg);
            using var child = Process.Start(start);
            if (child is null)
                return null;
            pid = child.Id;
        }
        return pid;
    }

    internal static string QuoteWindowsArgument(string value)
    {
        var result = new StringBuilder("\"");
        var slashes = 0;
        foreach (var c in value)
        {
            if (c == '\\')
            {
                slashes++;
                continue;
            }
            result.Append('\\', c == '"' ? slashes * 2 + 1 : slashes).Append(c);
            slashes = 0;
        }
        return result.Append('\\', slashes * 2).Append('"').ToString();
    }
}
