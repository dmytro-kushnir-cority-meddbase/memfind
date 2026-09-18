using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Microsoft.ML.Tokenizers;

namespace MemFind;

internal sealed class Embedder : IDisposable
{
    public const string ModelName = "bge-small-en-v1.5";
    public const int Dim = 384;
    public const string QueryPrefix = "Represent this sentence for searching relevant passages: ";

    private const int MaxTokens = 512;
    private const string ModelUrl = "https://huggingface.co/BAAI/bge-small-en-v1.5/resolve/main/onnx/model.onnx";
    private const string VocabUrl = "https://huggingface.co/BAAI/bge-small-en-v1.5/resolve/main/vocab.txt";

    private readonly InferenceSession _session;
    private readonly BertTokenizer _tokenizer;
    private readonly string[] _inputNames;
    private readonly string _outputName;
    private readonly int _sepId;

    private Embedder(InferenceSession session, BertTokenizer tokenizer)
    {
        _session = session;
        _tokenizer = tokenizer;

        var wanted = new[] { "input_ids", "attention_mask", "token_type_ids" };
        _inputNames = wanted.Where(n => session.InputMetadata.ContainsKey(n)).ToArray();
        if (_inputNames.Length == 0)
            throw new InvalidOperationException("model exposes none of input_ids/attention_mask/token_type_ids");

        _outputName = session.OutputMetadata.ContainsKey("last_hidden_state")
            ? "last_hidden_state"
            : session.OutputMetadata.Keys.First();

        var sep = tokenizer.EncodeToIds("", addSpecialTokens: true);
        _sepId = sep.Count > 0 ? sep[^1] : 102;
    }

    public static Embedder Load()
    {
        var dir = Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
            "memfind",
            ModelName);
        Directory.CreateDirectory(dir);

        var modelPath = Path.Combine(dir, "model.onnx");
        var vocabPath = Path.Combine(dir, "vocab.txt");

        EnsureFile(modelPath, ModelUrl);
        EnsureFile(vocabPath, VocabUrl);

        var options = new SessionOptions();
        options.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;

        var session = new InferenceSession(modelPath, options);
        var tokenizer = BertTokenizer.Create(vocabPath);
        return new Embedder(session, tokenizer);
    }

    private static void EnsureFile(string path, string url)
    {
        if (File.Exists(path) && new FileInfo(path).Length > 0) return;

        Console.Error.WriteLine($"downloading {Path.GetFileName(path)} from {url}");

        var tmp = path + ".tmp";
        if (File.Exists(tmp)) File.Delete(tmp);

        using var handler = new HttpClientHandler { AllowAutoRedirect = true };
        using var http = new HttpClient(handler) { Timeout = TimeSpan.FromMinutes(20) };

        using var response = http.Send(
            new HttpRequestMessage(HttpMethod.Get, url),
            HttpCompletionOption.ResponseHeadersRead);
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

    public float[] EmbedQuery(string query) => Embed(QueryPrefix + query);

    public float[] EmbedDocument(string text) => Embed(text);

    private float[] Embed(string text)
    {
        var ids = _tokenizer.EncodeToIds(text, addSpecialTokens: true);
        var n = Math.Min(ids.Count, MaxTokens);

        var inputIds = new long[n];
        for (var i = 0; i < n; i++) inputIds[i] = ids[i];
        if (ids.Count > MaxTokens) inputIds[n - 1] = _sepId;

        var dims = new[] { 1, n };
        var inputs = new List<NamedOnnxValue>(_inputNames.Length);

        foreach (var name in _inputNames)
        {
            var data = new long[n];
            switch (name)
            {
                case "input_ids":
                    Array.Copy(inputIds, data, n);
                    break;
                case "attention_mask":
                    Array.Fill(data, 1L);
                    break;
                default:
                    break;
            }
            inputs.Add(NamedOnnxValue.CreateFromTensor(name, new DenseTensor<long>(data, dims)));
        }

        using var results = _session.Run(inputs, new[] { _outputName });
        var tensor = results.First().AsTensor<float>();

        var vec = new float[Dim];
        for (var i = 0; i < Dim; i++) vec[i] = tensor[0, 0, i];

        Normalize(vec);
        return vec;
    }

    private static void Normalize(float[] v)
    {
        double sum = 0;
        foreach (var x in v) sum += (double)x * x;
        var norm = Math.Sqrt(sum);
        if (norm <= 0) return;
        for (var i = 0; i < v.Length; i++) v[i] = (float)(v[i] / norm);
    }

    public void Dispose() => _session.Dispose();
}
