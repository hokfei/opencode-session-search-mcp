# opencode-session-search-mcp

An MCP server that searches your past [OpenCode](https://opencode.ai) conversations by meaning **and** by exact keyword. It replaces the Python `mcp-session-search`.

Everything runs locally: embeddings, reranking and the optional answer model use [transformers.js](https://github.com/huggingface/transformers.js) on the CPU. No API keys, no network after the first model download.

## What it does

- **Hybrid search.** Semantic vectors (bge-small) and SQLite FTS5 keyword search are fused with reciprocal rank fusion. Ticket ids, error codes and file names (`GAT-1234`) match exactly; paraphrases match by meaning.
- **Swappable embedding model.** Set `SESSION_SEARCH_EMBED_MODEL`. The index rebuilds itself when the model changes.
- **Optional cross-encoder reranker.** Reorders the top candidates for better precision.
- **Optional answer model.** Writes a short answer from the top matches.
- **Exact vector scan over SQLite BLOBs.** Brute force, not an ANN index: no native dependencies, and fine up to roughly 100k chunks.
- **Reads sessions through the OpenCode client API**, not the SQLite file, so schema changes in OpenCode don't break it.

## Install

Requires Node >= 22.13 and a running OpenCode.

In `opencode.json`:

```json
{
  "mcp": {
    "servers": {
      "session-search": {
        "type": "local",
        "command": ["npx", "-y", "github:hokfei/opencode-session-search-mcp"],
        "environment": { "SESSION_SEARCH_RERANK_MODEL": "Xenova/bge-reranker-base" }
      }
    }
  }
}
```

The `environment` line is optional. This config layout follows my other OpenCode MCP repos and is not verified against a live OpenCode.

The package installs about 560 MB of dependencies, so the first `npx` run is slow. The first search downloads models: bge-small is about 34 MB; the reranker about 280 MB; Qwen2.5-1.5B about 1 GB. They are cached afterwards.

## Tools

| Tool | Purpose |
| --- | --- |
| `search_sessions` | `query`, optional `limit`, `answer`. Indexes new sessions first (bounded by `AUTO_INDEX_LIMIT`). |
| `get_session` | Read one transcript, truncated to `maxChars` (default 20000). |
| `list_sessions` | Recently updated indexed sessions. |
| `index_sessions` | Index new or changed sessions now. |
| `rebuild_index` | Wipe and re-embed everything (`confirm: true`). |
| `health_check` | Index size, active models, whether OpenCode is reachable. |

## Configuration

All variables are prefixed `SESSION_SEARCH_`. Set them in the MCP entry's `environment`.

| Variable | Default | Meaning |
| --- | --- | --- |
| `DB_PATH` | `~/.opencode-session-search/index.db` | Index location. |
| `EMBED_MODEL` | `Xenova/bge-small-en-v1.5` | Embedding model. |
| `EMBED_POOLING` / `EMBED_DTYPE` | from preset / `q8` | Override pooling (`cls`, `mean`) or quantization. |
| `QUERY_PREFIX` / `DOCUMENT_PREFIX` | from preset | Prefixes some models need. |
| `RERANK_MODEL` | off | e.g. `Xenova/bge-reranker-base`. |
| `RERANK_TOP` / `RERANK_MIN_SCORE` | `20` / off | Candidates reranked; drop results scoring below this. |
| `ANSWER_MODEL` | off | e.g. `onnx-community/Qwen2.5-1.5B-Instruct`. |
| `SIMILARITY_THRESHOLD` | `0.6` | Minimum cosine for vector hits. |
| `DEFAULT_TOP_K` / `MAX_TOP_K` | `5` / `20` | Result counts. |
| `CANDIDATES` | `50` | Candidates per signal before fusion. |
| `CHUNK_CHARS` / `CHUNK_OVERLAP` | `1200` / `200` | Chunking. |
| `INCLUDE_ARCHIVED` / `INCLUDE_SUBAGENTS` | `false` | Index those sessions too. |
| `AUTO_INDEX` / `AUTO_INDEX_LIMIT` | `true` / `200` | Index before each search. |

Presets exist for `Xenova/bge-small-en-v1.5`, `Xenova/bge-base-en-v1.5`, `Xenova/all-MiniLM-L6-v2` and `nomic-ai/nomic-embed-text-v1.5`.

## What is verified, and what is not

Verified:

- Typecheck, 34 unit and in-process MCP tests.
- Real `bge-small`: a paraphrase query, an exact `GAT-2025` keyword query and an unanswerable query behave correctly on a small fixture.
- Real `bge-reranker-base`: ranks relevant passages above irrelevant ones on that fixture.
- Real `Qwen2.5-1.5B-Instruct`: answered a VRAM question correctly (about 6 s on CPU). `Qwen2.5-0.5B-Instruct` ran but gave a wrong answer, so it is not recommended.
- A packed tarball installs and speaks MCP over stdio. Without OpenCode it returns a clear error.

Not verified:

- **Against a live OpenCode service.** The client calls follow the published types but have not run against real data.
- Embedding presets other than bge-small.
- Quality on a large, real history. The `0.6` threshold is carried over from the Python version and should be recalibrated, especially with a reranker or a different model.
- Reranker score scale: relevant passages scored positive and irrelevant ones negative in a few examples only. Tune `RERANK_MIN_SCORE` on real data.

Known weakness: keyword hits are not thresholded, so a query with common words can return loosely related sessions. The reranker with `RERANK_MIN_SCORE` is the mitigation.

## Migrating from `mcp-session-search`

The index is new (different path and format) and is built on first use. Nothing is read from the old index.

## Development

```sh
npm install
npm run typecheck
npm test
npm run build          # dist/ is committed so `npx github:...` works
node scripts/smoke.ts [rerank]   # manual check with the real models
node scripts/answer.ts [model]   # manual check of an answer model
```

Rebuild and commit `dist/` before every push.

## License

MIT
