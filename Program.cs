using MemFind;

var store = Store.DefaultRoot;
var k = 5;
var words = new List<string>();

for (var i = 0; i < args.Length; i++)
{
    switch (args[i])
    {
        case "--help" or "-h" or "/?":
            Usage();
            return 0;

        case "--store":
            if (i + 1 >= args.Length) { Console.Error.WriteLine("--store needs a path"); return 2; }
            store = args[++i];
            break;

        case "-k":
            if (i + 1 >= args.Length || !int.TryParse(args[++i], out k) || k < 1)
            {
                Console.Error.WriteLine("-k needs a positive integer");
                return 2;
            }
            break;

        default:
            words.Add(args[i]);
            break;
    }
}

if (words.Count == 0)
{
    Usage();
    return 0;
}

if (words.Count == 1 && words[0] == "index") return MemFind.Index.Build(store);

return MemFind.Index.Search(store, string.Join(' ', words), k);

static void Usage()
{
    Console.WriteLine("""
        memfind — semantic search over the Claude Code memory store

          memfind index [--store <path>]       build the index (<store>\.memfind\index.json)
          memfind <query words...> [-k N]      search; default k=5
          memfind --help                       this screen

        Options:
          --store <path>   memory store root (default: C:\Git\docs\memory)
          -k N             number of hits to print (default: 5)

        Model: BAAI/bge-small-en-v1.5 (384-dim), cached in %LOCALAPPDATA%\memfind\,
        downloaded from huggingface.co on first use.

        Examples:
          memfind index
          memfind how do I switch which clone IIS serves
          memfind "unsigned installer" -k 10
        """);
}
