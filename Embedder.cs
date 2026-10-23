using System.Runtime.InteropServices;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Microsoft.ML.Tokenizers;

namespace MemFind;

internal enum Pooling
{
    Cls,
    Pooled,
}

internal enum TokenizerKind
{
    WordPiece,
    SentencePiece,
}

/// <summary>
/// One embedding model: what to fetch, how to tokenize, which output tensor carries the embedding.
/// Read from here, never probed off the session — a model that also exposes an unpooled tensor
/// would otherwise be read at the wrong shape and return plausible garbage.
/// </summary>
internal sealed record ModelSpec(
    string Key,
    string Name,
    string Suffix,
    int Dim,
    int MaxTokens,
    int ChatMaxTokens,
    string QueryPrefix,
    string DocPrefix,
    Pooling Pooling,
    TokenizerKind Tokenizer,
    string TokenizerFile,
    string TokenizerSibling,
    int PadId,
    int SepId,
    string OutputName,
    string ModelFile,
    (string Url, string File)[] Files,
    double Floor
)
{
    public static readonly ModelSpec Bge = new(
        Key: "bge",
        Name: "bge-small-en-v1.5-int8",
        Suffix: "",
        Dim: 384,
        MaxTokens: 512,
        ChatMaxTokens: 256,
        QueryPrefix: "Represent this sentence for searching relevant passages: ",
        DocPrefix: "",
        Pooling: Pooling.Cls,
        Tokenizer: TokenizerKind.WordPiece,
        TokenizerFile: "vocab.txt",
        TokenizerSibling: "bge-small-en-v1.5",
        PadId: 0,
        SepId: 102,
        OutputName: "last_hidden_state",
        ModelFile: "model.onnx",
        Files:
        [
            (
                "https://huggingface.co/Xenova/bge-small-en-v1.5/resolve/main/onnx/model_quantized.onnx",
                "model.onnx"
            ),
            ("https://huggingface.co/Xenova/bge-small-en-v1.5/resolve/main/vocab.txt", "vocab.txt"),
        ],
        Floor: 0.70
    );

    public static readonly ModelSpec Gemma = new(
        Key: "gemma",
        Name: "embeddinggemma-300m-int8",
        Suffix: ".gemma",
        Dim: 768,
        MaxTokens: 512,
        ChatMaxTokens: 256,
        QueryPrefix: "task: search result | query: ",
        DocPrefix: "title: none | text: ",
        Pooling: Pooling.Pooled,
        Tokenizer: TokenizerKind.SentencePiece,
        TokenizerFile: "tokenizer.model",
        TokenizerSibling: "",
        PadId: 0,
        SepId: 1,
        OutputName: "sentence_embedding",
        ModelFile: "model_quantized.onnx",
        Files:
        [
            (
                "https://huggingface.co/onnx-community/embeddinggemma-300m-ONNX/resolve/main/onnx/model_quantized.onnx",
                "model_quantized.onnx"
            ),
            // ORT resolves the external weights by the name in the graph, so both files keep
            // their published names and live side by side.
            (
                "https://huggingface.co/onnx-community/embeddinggemma-300m-ONNX/resolve/main/onnx/model_quantized.onnx_data",
                "model_quantized.onnx_data"
            ),
            (
                "https://huggingface.co/onnx-community/embeddinggemma-300m-ONNX/resolve/main/tokenizer.model",
                "tokenizer.model"
            ),
        ],
        Floor: 0.38
    );

    public static readonly ModelSpec Default = Bge;

    /// <summary>Every model, in CLI order.</summary>
    public static readonly ModelSpec[] All = [Bge, Gemma];

    /// <summary>One per distinct set of index files on disk; bge and bge-fp32 share theirs.</summary>
    public static readonly ModelSpec[] Indexes = [Bge, Gemma];

    public static string Keys => string.Join('|', All.Select(m => m.Key));

    public static ModelSpec? ByKey(string key) => All.FirstOrDefault(m => m.Key == key);

    public static ModelSpec? ByName(string name) => All.FirstOrDefault(m => m.Name == name);
}

internal sealed class Embedder : IDisposable
{
    public static bool AllowDownloads { get; set; } = true;
    public const int BatchSize = 64;

    private static int _threads = Math.Max(1, Environment.ProcessorCount / 2);
    private static int _queryThreads = 4;

    /// <summary>Intra-op threads per session and the degree of parallelism used while parsing.</summary>
    public static int Threads
    {
        get => _threads;
        set => _threads = _queryThreads = Math.Max(1, value);
    }

    /// <summary>
    /// BertTokenizer hides the base truncating overload with one that adds [CLS]/[SEP], so the two
    /// tokenizers are bound at their own type and reached through this, never through Tokenizer.
    /// </summary>
    private delegate IReadOnlyList<int> Encoder(string text, int maxTokens, out int charsConsumed);

    private readonly InferenceSession _session;
    private readonly ThreadLocal<Encoder> _tokenizers;
    private readonly string[] _inputNames;
    private readonly string _outputName;
    private readonly int _sepId;
    private readonly int _padId;
    private string? _query;
    private float[] _queryVector = [];

    public ModelSpec Spec { get; }
    public string Model => Spec.Name;
    public int Dim => Spec.Dim;
    public int IntraOpThreads { get; }

    private Embedder(ModelSpec spec, InferenceSession session, int threads, string tokenizerPath)
    {
        Spec = spec;
        _session = session;
        IntraOpThreads = threads;
        _tokenizers = new ThreadLocal<Encoder>(
            () =>
            {
                if (spec.Tokenizer == TokenizerKind.WordPiece)
                {
                    var bert = BertTokenizer.Create(tokenizerPath);
                    return (string text, int maxTokens, out int charsConsumed) =>
                        bert.EncodeToIds(text, maxTokens, out _, out charsConsumed);
                }

                using var stream = File.OpenRead(tokenizerPath);
                var sentencePiece = SentencePieceTokenizer.Create(
                    stream,
                    addBeginningOfSentence: true,
                    addEndOfSentence: false
                );
                return (string text, int maxTokens, out int charsConsumed) =>
                    sentencePiece.EncodeToIds(text, maxTokens, out _, out charsConsumed);
            },
            trackAllValues: false
        );

        var wanted = new[] { "input_ids", "attention_mask", "token_type_ids" };
        _inputNames = wanted.Where(n => session.InputMetadata.ContainsKey(n)).ToArray();
        if (_inputNames.Length == 0)
            throw new InvalidOperationException(
                "model exposes none of input_ids/attention_mask/token_type_ids"
            );

        _outputName = spec.OutputName;
        _sepId = spec.SepId;
        _padId = spec.PadId;
    }

    /// <summary>Starts a background load of the query session for <paramref name="spec"/>, once.</summary>
    public static void Preload(Dictionary<ModelSpec, Task<Embedder>> embedders, ModelSpec spec)
    {
        if (!embedders.ContainsKey(spec))
            embedders[spec] = Task.Run(() => Load(spec, query: true));
    }

    /// <summary>The query session for <paramref name="spec"/>: the preloaded one, else loaded now.</summary>
    public static Embedder ForQuery(Dictionary<ModelSpec, Task<Embedder>> embedders, ModelSpec spec)
    {
        if (!embedders.TryGetValue(spec, out var loading))
            embedders[spec] = loading = Task.FromResult(Load(spec, query: true));
        return loading.GetAwaiter().GetResult();
    }

    /// <summary>
    /// Waits for every load, used or not, so none is still inside ONNX Runtime at exit. A load
    /// nothing used may fail silently; one a search used has already thrown there.
    /// </summary>
    public static void DisposeAll(Dictionary<ModelSpec, Task<Embedder>> embedders)
    {
        foreach (var loading in embedders.Values)
        {
            try
            {
                loading.GetAwaiter().GetResult().Dispose();
            }
            catch (Exception) when (loading.IsFaulted) { }
        }
    }

    public static Embedder Load(ModelSpec spec, bool gpu = false, bool query = false)
    {
        var root = Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
            "memfind"
        );

        var dir = Path.Combine(root, spec.Name);
        Directory.CreateDirectory(dir);

        foreach (var (url, name) in spec.Files)
        {
            var path = Path.Combine(dir, name);
            if (name == spec.TokenizerFile && spec.TokenizerSibling.Length > 0)
                EnsureVocab(path, Path.Combine(root, spec.TokenizerSibling, name), url);
            else
                EnsureFile(path, url);
        }

        var threads = query ? _queryThreads : Threads;

        // Keep the options alive throughout native initialization, including background loads.
        // NativeAOT can otherwise collect them while another model's index is being decoded.
        using var options = new SessionOptions();
        options.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;
        options.IntraOpNumThreads = threads;
        options.InterOpNumThreads = 1;
        if (query)
            options.AddSessionConfigEntry("session.disable_prepacking", "1");

        if (gpu && TryEnableCuda(options))
            Console.Error.WriteLine("cuda enabled");

        var model = Path.Combine(dir, spec.ModelFile);
        return new Embedder(
            spec,
            query && !gpu && !spec.Files.Any(f => f.File.EndsWith(".onnx_data"))
                ? LoadOptimized(model, options)
                : new InferenceSession(model, options),
            threads,
            Path.Combine(dir, spec.TokenizerFile)
        );
    }

    /// <summary>
    /// Loads model.optimized.onnx when model.optimized.key matches, else optimises the model and
    /// saves both. memfind-rs reads and writes the same two files.
    /// </summary>
    private static InferenceSession LoadOptimized(string model, SessionOptions options)
    {
        var graph = Path.Combine(Path.GetDirectoryName(model)!, "model.optimized.onnx");
        var keyPath = Path.ChangeExtension(graph, ".key");
        var source = new FileInfo(model);
        var key =
            $"{source.Length} bytes, modified {(source.LastWriteTimeUtc - DateTime.UnixEpoch).Ticks * 100} ns, {BuildInfo()}";

        try
        {
            if (File.Exists(keyPath) && File.Exists(graph) && File.ReadAllText(keyPath) == key)
            {
                options.GraphOptimizationLevel = GraphOptimizationLevel.ORT_DISABLE_ALL;
                return new InferenceSession(graph, options);
            }
        }
        catch (Exception e) when (e is IOException or UnauthorizedAccessException) { }

        var partial = Path.ChangeExtension(graph, $".{Environment.ProcessId}.tmp");
        options.OptimizedModelFilePath = partial;
        // ORT warns that a graph saved at this level is tied to this CPU; the cache never leaves it.
        options.LogSeverityLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_ERROR;
        InferenceSession session;
        try
        {
            session = new InferenceSession(model, options);
        }
        catch (Exception)
        {
            // An unwritable cache fails the session, and ORT codes the managed wrapper cannot map
            // surface as KeyNotFoundException; a real model error recurs on this second load.
            options.OptimizedModelFilePath = "";
            return new InferenceSession(model, options);
        }

        try
        {
            File.Move(partial, graph, overwrite: true);
            File.WriteAllText(keyPath, key);
        }
        catch (Exception e) when (e is IOException or UnauthorizedAccessException)
        {
            try
            {
                File.Delete(partial);
            }
            catch (Exception f) when (f is IOException or UnauthorizedAccessException) { }
        }

        return session;
    }

    [DllImport("onnxruntime")]
    private static extern nint OrtGetApiBase();

    private static unsafe string BuildInfo()
    {
        // OrtApiBase.GetApi, then OrtApi slot 254 = GetBuildInfoString (onnxruntime_c_api.h, since
        // API 15); the managed API does not expose it.
        var getApi = *(delegate* unmanaged<uint, nint*>*)OrtGetApiBase();
        var api = getApi(15);
        return Marshal.PtrToStringUTF8(((delegate* unmanaged<nint>)api[254])())!;
    }

    /// <summary>
    /// cuDNN ships separately from the ORT GPU package, and a missing one surfaces as a
    /// NOT_IMPLEMENTED on the first Run rather than at session load — too late to fall back. So
    /// both directories are found and put on PATH up front, or the CUDA provider is never appended.
    /// </summary>
    private static bool TryEnableCuda(SessionOptions options)
    {
        if (!OperatingSystem.IsWindows())
        {
            Console.Error.WriteLine("cuda auto-detection is only supported on Windows, using cpu");
            return false;
        }

        var cudnn = FindDirectory(
            Environment.GetEnvironmentVariable("MEMFIND_CUDNN"),
            "cudnn64_9.dll",
            [
                Path.Combine(
                    Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
                    "memfind",
                    "cudnn"
                ),
                @"C:\Tools",
            ]
        );

        var cuda = FindDirectory(
            Environment.GetEnvironmentVariable("MEMFIND_CUDA"),
            "cudart64_13.dll",
            [@"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"]
        );

        if (cudnn is null || cuda is null)
        {
            Console.Error.WriteLine(
                $"cuda unavailable ({(cudnn is null ? "cudnn64_9.dll" : "cudart64_13.dll")} not found), using cpu"
            );
            return false;
        }

        Environment.SetEnvironmentVariable(
            "PATH",
            $"{cudnn};{cuda};{Environment.GetEnvironmentVariable("PATH")}"
        );

        try
        {
            options.AppendExecutionProvider_CUDA();
            // The CUDA EP warns on every session that shape ops stayed on the CPU, which is normal.
            options.LogSeverityLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_ERROR;
            return true;
        }
        catch (Exception e)
        {
            Console.Error.WriteLine($"cuda unavailable ({e.GetType().Name}), using cpu");
            return false;
        }
    }

    private static string? FindDirectory(string? preferred, string marker, string[] roots)
    {
        if (preferred is not null && File.Exists(Path.Combine(preferred, marker)))
            return preferred;

        foreach (var root in roots)
        {
            if (File.Exists(Path.Combine(root, marker)))
                return root;
            if (!Directory.Exists(root))
                continue;
            var hit = Directory
                .EnumerateFiles(root, marker, SearchOption.AllDirectories)
                .OrderDescending()
                .FirstOrDefault();
            if (hit is not null)
                return Path.GetDirectoryName(hit);
        }

        return null;
    }

    private static void EnsureVocab(string path, string sibling, string url)
    {
        if (File.Exists(path) && new FileInfo(path).Length > 0)
            return;

        if (File.Exists(sibling) && new FileInfo(sibling).Length > 0)
        {
            File.Copy(sibling, path);
            return;
        }

        EnsureFile(path, url);
    }

    private static void EnsureFile(string path, string url)
    {
        if (File.Exists(path) && new FileInfo(path).Length > 0)
            return;

        if (!AllowDownloads)
            throw new FileNotFoundException("model file unavailable in offline mode", path);

        Console.Error.WriteLine($"downloading {Path.GetFileName(path)} from {url}");

        var tmp = path + ".tmp";
        if (File.Exists(tmp))
            File.Delete(tmp);

        using var handler = new HttpClientHandler { AllowAutoRedirect = true };
        using var http = new HttpClient(handler) { Timeout = TimeSpan.FromMinutes(20) };

        using var response = http.Send(
            new HttpRequestMessage(HttpMethod.Get, url),
            HttpCompletionOption.ResponseHeadersRead
        );
        response.EnsureSuccessStatusCode();

        var total = response.Content.Headers.ContentLength ?? -1L;

        using (var source = response.Content.ReadAsStream())
        using (var target = File.Create(tmp))
        {
            var buffer = new byte[81920];
            long read = 0;
            long lastReport = 0;
            int n;
            while ((n = source.Read(buffer, 0, buffer.Length)) > 0)
            {
                target.Write(buffer, 0, n);
                read += n;
                if (read - lastReport >= 8 * 1024 * 1024)
                {
                    lastReport = read;
                    if (total > 0)
                        Console.Error.WriteLine($"  {read / 1048576} MB / {total / 1048576} MB");
                    else
                        Console.Error.WriteLine($"  {read / 1048576} MB");
                }
            }
        }

        File.Move(tmp, path, overwrite: true);
        Console.Error.WriteLine($"  saved {path}");
    }

    public int[] Encode(string text, int maxTokens)
    {
        var encode = _tokenizers.Value!;
        var ids = encode(text, maxTokens, out var charsConsumed);

        var result = new int[ids.Count];
        for (var i = 0; i < result.Length; i++)
            result[i] = ids[i];
        if (result.Length == maxTokens && charsConsumed < text.Length)
            result[^1] = _sepId;
        return result;
    }

    public float[] EmbedQuery(string query)
    {
        if (query != _query)
        {
            _queryVector = Embed(Spec.QueryPrefix + query);
            _query = query;
        }
        return _queryVector;
    }

    public float[] EmbedDocument(string text) => Embed(Spec.DocPrefix + text);

    private float[] Embed(string text)
    {
        var tokens = new[] { Encode(text, Spec.MaxTokens) };
        var vectors = new float[Dim];
        RunBatch(_session, tokens, 0, 1, vectors, 0);
        return vectors;
    }

    /// <summary>Embeds every token sequence into <paramref name="destination"/> at index * Dim.</summary>
    public void EmbedMany(
        IReadOnlyList<int[]> tokens,
        float[] destination,
        Action<int, long>? progress = null,
        IReadOnlyList<int>? characters = null
    )
    {
        var order = new int[tokens.Count];
        for (var i = 0; i < order.Length; i++)
            order[i] = i;
        Array.Sort(order, (a, b) => tokens[a].Length.CompareTo(tokens[b].Length));

        var batch = new int[BatchSize][];
        var pooled = new float[BatchSize * Dim];
        var done = 0;
        long chars = 0;

        for (var from = 0; from < order.Length; from += BatchSize)
        {
            var count = Math.Min(BatchSize, order.Length - from);
            for (var i = 0; i < count; i++)
                batch[i] = tokens[order[from + i]];

            RunBatch(_session, batch, 0, count, pooled, 0);

            for (var i = 0; i < count; i++)
                Array.Copy(pooled, (long)i * Dim, destination, (long)order[from + i] * Dim, Dim);

            done += count;
            if (characters is not null)
            {
                for (var i = 0; i < count; i++)
                {
                    chars += characters[order[from + i]];
                }
            }
            progress?.Invoke(done, chars);
        }
    }

    private void RunBatch(
        InferenceSession session,
        IReadOnlyList<int[]> batch,
        int offset,
        int count,
        float[] destination,
        int destinationIndex
    )
    {
        var maxLen = 1;
        for (var i = 0; i < count; i++)
            maxLen = Math.Max(maxLen, batch[offset + i].Length);

        var dims = new[] { count, maxLen };
        var inputs = new List<NamedOnnxValue>(_inputNames.Length);

        foreach (var name in _inputNames)
        {
            var data = new long[count * maxLen];
            switch (name)
            {
                case "input_ids":
                    for (var i = 0; i < count; i++)
                    {
                        var ids = batch[offset + i];
                        var row = i * maxLen;
                        for (var j = 0; j < maxLen; j++)
                            data[row + j] = j < ids.Length ? ids[j] : _padId;
                    }
                    break;
                case "attention_mask":
                    for (var i = 0; i < count; i++)
                    {
                        var len = batch[offset + i].Length;
                        var row = i * maxLen;
                        for (var j = 0; j < len; j++)
                            data[row + j] = 1L;
                    }
                    break;
                default:
                    break;
            }
            inputs.Add(NamedOnnxValue.CreateFromTensor(name, new DenseTensor<long>(data, dims)));
        }

        using var results = session.Run(inputs, new[] { _outputName });
        var tensor = results.First().AsTensor<float>();

        // Cls reads row 0 of a [batch, seq, dim] tensor; Pooled is already [batch, dim].
        var stride = Spec.Pooling == Pooling.Cls ? maxLen * Dim : Dim;

        if (tensor is DenseTensor<float> dense)
        {
            var span = dense.Buffer.Span;
            for (var i = 0; i < count; i++)
            {
                var row = span.Slice(i * stride, Dim);
                row.CopyTo(destination.AsSpan((destinationIndex + i) * Dim, Dim));
            }
        }
        else
        {
            for (var i = 0; i < count; i++)
            for (var j = 0; j < Dim; j++)
                destination[(destinationIndex + i) * Dim + j] =
                    Spec.Pooling == Pooling.Cls ? tensor[i, 0, j] : tensor[i, j];
        }

        for (var i = 0; i < count; i++)
            Normalize(destination.AsSpan((destinationIndex + i) * Dim, Dim));
    }

    private static void Normalize(Span<float> v)
    {
        double sum = 0;
        foreach (var x in v)
            sum += (double)x * x;
        var norm = Math.Sqrt(sum);
        if (norm <= 0)
            return;
        for (var i = 0; i < v.Length; i++)
            v[i] = (float)(v[i] / norm);
    }

    public void Dispose()
    {
        _session.Dispose();
        _tokenizers.Dispose();
    }
}
