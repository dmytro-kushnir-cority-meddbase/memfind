using System.Text.Json.Serialization;

namespace MemFind;

[JsonSourceGenerationOptions(WriteIndented = false)]
[JsonSerializable(typeof(IndexFile))]
[JsonSerializable(typeof(ChatState))]
internal sealed partial class Ctx : JsonSerializerContext;
