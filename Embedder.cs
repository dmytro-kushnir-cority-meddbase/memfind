using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Microsoft.ML.Tokenizers;

namespace MemFind;

internal sealed class Embedder : IDisposable
{
    public const string Fp32ModelName = "bge-small-en-v1.5";
    public const string Int8ModelName = "bge-small-en-v1.5-int8";
    public const string ModelName = Fp32ModelName;
    public const int Dim = 384;
    public const string QueryPrefix = "Represent this sentence for searching relevant passages: ";

    public const int MaxTokens = 512;
    public const int ChatMaxTokens = 256;
    public const int BatchSize = 64;

    private const string Fp32ModelUrl =
        "https://huggingface.co/BAAI/bge-small-en-v1.5/resolve/main/onnx/model.onnx";
    private const string Fp32VocabUrl =
        "https://huggingface.co/BAAI/bge-small-en-v1.5/resolve/main/vocab.txt";
    private const string Int8ModelUrl =
        "https://huggingface.co/Xenova/bge-small-en-v1.5/resolve/main/onnx/model_quantized.onnx";
    private const string Int8VocabUrl =
        "https://huggingface.co/Xenova/bge-small-en-v1.5/resolve/main/vocab.txt";

    private static int _threads = Math.Max(1, Environment.ProcessorCount / 2);

    /// <summary>Intra-op threads per session and the degree of parallelism used while parsing.</summary>
    public static int Threads
    {
        get => _threads;
        set => _threads = Math.Max(1, value);
    }

    private readonly InferenceSession _session;
    private readonly ThreadLocal<BertTokenizer> _tokenizers;
    private readonly string[] _inputNames;
    private readonly string _outputName;
    private readonly int _sepId;
    private readonly int _padId;

    public string Model { get; }
    public int IntraOpThreads { get; }

    private Embedder(string model, InferenceSession session, int threads, string vocabPath)
    {
        Model = model;
        _session = session;
        IntraOpThreads = threads;
        _tokenizers = new ThreadLocal<BertTokenizer>(
            () => BertTokenizer.Create(vocabPath),
            trackAllValues: false
        );

        var wanted = new[] { "input_ids", "attention_mask", "token_type_ids" };
        _inputNames = wanted.Where(n => session.InputMetadata.ContainsKey(n)).ToArray();
        if (_inputNames.Length == 0)
            throw new InvalidOperationException(
                "model exposes none of input_ids/attention_mask/token_type_ids"
            );

        _outputName = session.OutputMetadata.ContainsKey("last_hidden_state")
            ? "last_hidden_state"
            : session.OutputMetadata.Keys.First();

        var tokenizer = _tokenizers.Value!;
        _sepId = tokenizer.SeparatorTokenId;
        _padId = tokenizer.PaddingTokenId;
    }

    public static Embedder Load(bool int8 = true)
    {
        var model = int8 ? Int8ModelName : Fp32ModelName;
        var root = Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
            "memfind"
        );

        var dir = Path.Combine(root, model);
        Directory.CreateDirectory(dir);

        var modelPath = Path.Combine(dir, "model.onnx");
        var vocabPath = Path.Combine(dir, "vocab.txt");

        EnsureFile(modelPath, int8 ? Int8ModelUrl : Fp32ModelUrl);
        EnsureVocab(
            vocabPath,
            Path.Combine(root, int8 ? Fp32ModelName : Int8ModelName, "vocab.txt"),
            int8 ? Int8VocabUrl : Fp32VocabUrl
        );

        var threads = Threads;

        var options = new SessionOptions();
        options.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;
        options.IntraOpNumThreads = threads;
        options.InterOpNumThreads = 1;

        return new Embedder(model, new InferenceSession(modelPath, options), threads, vocabPath);
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
        var tokenizer = _tokenizers.Value!;
        var ids = tokenizer.EncodeToIds(text, maxTokens, out _, out var charsConsumed);

        var result = new int[ids.Count];
        for (var i = 0; i < result.Length; i++)
            result[i] = ids[i];
        if (result.Length == maxTokens && charsConsumed < text.Length)
            result[^1] = _sepId;
        return result;
    }

    public float[] EmbedQuery(string query) => EmbedDocument(QueryPrefix + query);

    public float[] EmbedDocument(string text)
    {
        var tokens = new[] { Encode(text, MaxTokens) };
        var vectors = new float[Dim];
        RunBatch(_session, tokens, 0, 1, vectors, 0);
        return vectors;
    }

    /// <summary>Embeds every token sequence into <paramref name="destination"/> at index * Dim.</summary>
    public void EmbedMany(
        IReadOnlyList<int[]> tokens,
        float[] destination,
        Action<int>? progress = null
    )
    {
        var order = new int[tokens.Count];
        for (var i = 0; i < order.Length; i++)
            order[i] = i;
        Array.Sort(order, (a, b) => tokens[a].Length.CompareTo(tokens[b].Length));

        var batch = new int[BatchSize][];
        var pooled = new float[BatchSize * Dim];
        var done = 0;

        for (var from = 0; from < order.Length; from += BatchSize)
        {
            var count = Math.Min(BatchSize, order.Length - from);
            for (var i = 0; i < count; i++)
                batch[i] = tokens[order[from + i]];

            RunBatch(_session, batch, 0, count, pooled, 0);

            for (var i = 0; i < count; i++)
                Array.Copy(pooled, (long)i * Dim, destination, (long)order[from + i] * Dim, Dim);

            done += count;
            progress?.Invoke(done);
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

        if (tensor is DenseTensor<float> dense)
        {
            var span = dense.Buffer.Span;
            for (var i = 0; i < count; i++)
            {
                var cls = span.Slice(i * maxLen * Dim, Dim);
                cls.CopyTo(destination.AsSpan((destinationIndex + i) * Dim, Dim));
            }
        }
        else
        {
            for (var i = 0; i < count; i++)
            for (var j = 0; j < Dim; j++)
                destination[(destinationIndex + i) * Dim + j] = tensor[i, 0, j];
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
