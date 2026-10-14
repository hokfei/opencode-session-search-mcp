import type { Config } from "./config.ts";
import type { Store } from "./store.ts";
import type { Embedder, Reranker } from "./types.ts";

export interface Hit {
  readonly sessionId: string;
  readonly title: string;
  readonly directory?: string;
  readonly updated: number;
  readonly score: number;
  readonly vectorScore?: number;
  readonly keywordMatch: boolean;
  readonly reranked: boolean;
  readonly snippet: string;
}

export interface SearchDeps {
  readonly store: Store;
  readonly embedder: Embedder;
  readonly reranker?: Reranker;
  readonly config: Config;
}

const RRF_K = 60;
const SNIPPET_CHARS = 500;

/**
 * Hybrid search: vector similarity and keyword (bm25) rankings are fused with reciprocal
 * rank fusion, optionally reordered by a cross-encoder, then collapsed to the best chunk per session.
 * Vector hits must clear the similarity threshold; keyword hits are admitted by rank alone.
 */
export async function hybridSearch(deps: SearchDeps, query: string, topK: number): Promise<Hit[]> {
  const { store, embedder, reranker, config } = deps;
  const vector = store.vectorSearch(await embedder.embedQuery(query), config.candidates, config.threshold);
  const keyword = store.keywordSearch(query, config.candidates);

  const fused = new Map<number, number>();
  const vectorScores = new Map<number, number>();
  const keywordChunks = new Set<number>();
  vector.forEach((h, rank) => {
    fused.set(h.chunkId, (fused.get(h.chunkId) ?? 0) + 1 / (RRF_K + rank + 1));
    vectorScores.set(h.chunkId, h.score);
  });
  keyword.forEach((h, rank) => {
    fused.set(h.chunkId, (fused.get(h.chunkId) ?? 0) + 1 / (RRF_K + rank + 1));
    keywordChunks.add(h.chunkId);
  });

  let ranked = [...fused.entries()].sort((a, b) => b[1] - a[1]);
  const rows = store.chunks(ranked.map(([id]) => id));
  ranked = ranked.filter(([id]) => rows.has(id));

  let reranked = false;
  if (reranker && ranked.length) {
    const head = ranked.slice(0, config.rerankTop);
    const scores = await reranker.score(query, head.map(([id]) => rows.get(id)!.text));
    const reordered = head
      .map(([id], i) => [id, scores[i]!] as [number, number])
      .filter(([, s]) => config.rerankMinScore === undefined || s >= config.rerankMinScore)
      .sort((a, b) => b[1] - a[1]);
    ranked = reordered;
    reranked = true;
  }

  const seen = new Set<string>();
  const hits: Hit[] = [];
  for (const [chunkId, score] of ranked) {
    const row = rows.get(chunkId)!;
    if (seen.has(row.sessionId)) continue;
    seen.add(row.sessionId);
    const info = store.sessionInfo(row.sessionId);
    if (!info) continue;
    hits.push({
      sessionId: row.sessionId,
      title: info.title,
      directory: info.directory,
      updated: info.updated,
      score,
      vectorScore: vectorScores.get(chunkId),
      keywordMatch: keywordChunks.has(chunkId),
      reranked,
      snippet: row.text.length > SNIPPET_CHARS ? `${row.text.slice(0, SNIPPET_CHARS - 1)}…` : row.text,
    });
    if (hits.length >= topK) break;
  }
  return hits;
}
