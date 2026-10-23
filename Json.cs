using System.Text.Json.Serialization;

namespace MemFind;

[JsonSourceGenerationOptions(WriteIndented = false)]
[JsonSerializable(typeof(IndexFile))]
[JsonSerializable(typeof(ChatState))]
[JsonSerializable(typeof(MemoryHit))]
[JsonSerializable(typeof(ChatHit))]
[JsonSerializable(typeof(Show.Shown))]
internal sealed partial class Ctx : JsonSerializerContext;

/// <summary>One leg of a fused hit; scores in every hit record are rounded to 4 decimals.</summary>
internal sealed record HitLeg(
    [property: JsonPropertyName("model")] string Model,
    [property: JsonPropertyName("score")] double Score,
    [property: JsonPropertyName("rank")] int Rank
)
{
    public static List<HitLeg>? From(List<Index.Leg>? legs) =>
        legs?.Select(leg => new HitLeg(leg.Model, Math.Round(leg.Score, 4), leg.Rank)).ToList();
}

internal sealed record MemoryHit(
    [property: JsonPropertyName("source")] string Source,
    [property: JsonPropertyName("score")] double Score,
    [property: JsonPropertyName("file")] string File,
    [property: JsonPropertyName("line")] int Line,
    [property: JsonPropertyName("ts"), JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)] DateTimeOffset? Ts,
    [property: JsonPropertyName("theme")] string Theme,
    [property: JsonPropertyName("title")] string Title,
    [property: JsonPropertyName("description")] string Description,
    [property: JsonPropertyName("models"), JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)] List<HitLeg>? Models
);

internal sealed record ChatHit(
    [property: JsonPropertyName("source")] string Source,
    [property: JsonPropertyName("score")] double Score,
    [property: JsonPropertyName("file")] string File,
    [property: JsonPropertyName("line")] long Line,
    [property: JsonPropertyName("uuid")] string Uuid,
    [property: JsonPropertyName("chunk")] int Chunk,
    [property: JsonPropertyName("session")] string Session,
    [property: JsonPropertyName("cwd")] string Cwd,
    [property: JsonPropertyName("role")] string Role,
    [property: JsonPropertyName("ts")] string Ts,
    [property: JsonPropertyName("text")] string Text,
    [property: JsonPropertyName("models"), JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)] List<HitLeg>? Models
);
