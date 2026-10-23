using System.Buffers;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace MemFind;

[JsonSourceGenerationOptions(WriteIndented = false)]
[JsonSerializable(typeof(IndexFile))]
[JsonSerializable(typeof(ChatState))]
internal sealed partial class Ctx : JsonSerializerContext;

/// <summary>One JSON object per stdout line, written straight as UTF-8 (no reflection, AOT-safe).</summary>
internal sealed class JsonLines : IDisposable
{
    public static bool FullPrecision { get; set; }
    public static double Score(float score) => FullPrecision ? score : Math.Round(score, 4);
    private readonly Stream _stream = Console.OpenStandardOutput();
    private readonly ArrayBufferWriter<byte> _buffer = new(4096);
    private readonly Utf8JsonWriter _writer;

    public JsonLines()
    {
        _writer = new Utf8JsonWriter(_buffer, new JsonWriterOptions { SkipValidation = true });
    }

    public Utf8JsonWriter Writer => _writer;

    /// <summary>Ends the object started on <see cref="Writer"/> and emits it as one line.</summary>
    public void Flush()
    {
        _writer.Flush();
        _stream.Write(_buffer.WrittenSpan);
        _stream.WriteByte((byte)'\n');
        _buffer.Clear();
        _writer.Reset(_buffer);
    }

    public void Dispose()
    {
        // The stream is stdout itself; flush it but never close the handle — `--all` opens it twice.
        _writer.Dispose();
        _stream.Flush();
    }
}
