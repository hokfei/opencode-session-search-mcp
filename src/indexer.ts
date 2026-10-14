import { buildTranscript, chunkText } from "./chunking.ts";
import type { Config } from "./config.ts";
import type { Store } from "./store.ts";
import type { Embedder, SessionMeta, SessionSource } from "./types.ts";

export interface IndexReport {
  readonly scanned: number;
  readonly indexed: number;
  readonly unchanged: number;
  readonly removed: number;
  readonly remaining: number;
  readonly rebuilt: boolean;
  readonly failed: ReadonlyArray<{ readonly id: string; readonly error: string }>;
}

export interface IndexOptions {
  readonly force?: boolean;
  readonly limit?: number;
}

const BATCH = 16;

export class Indexer {
  readonly #source: () => Promise<SessionSource>;
  readonly #store: Store;
  readonly #embedder: Embedder;
  readonly #config: Config;
  #running?: Promise<IndexReport>;

  constructor(deps: { source: () => Promise<SessionSource>; store: Store; embedder: Embedder; config: Config }) {
    this.#source = deps.source;
    this.#store = deps.store;
    this.#embedder = deps.embedder;
    this.#config = deps.config;
  }

  /** Concurrent callers share one pass instead of racing each other. */
  run(options: IndexOptions = {}): Promise<IndexReport> {
    this.#running ??= this.#run(options).finally(() => {
      this.#running = undefined;
    });
    return this.#running;
  }

  async #run(options: IndexOptions): Promise<IndexReport> {
    const store = this.#store;
    const signature = this.#embedder.signature;
    const stored = store.getMeta("embedder");
    const rebuilt = Boolean(options.force) || (stored !== undefined && stored !== signature);
    if (rebuilt) store.clear();
    store.setMeta("embedder", signature);

    const source = await this.#source();
    const live: SessionMeta[] = [];
    for await (const meta of source.listSessions()) {
      if (meta.archived && !this.#config.includeArchived) continue;
      if (meta.subagent && !this.#config.includeSubagents) continue;
      live.push(meta);
    }

    const liveIds = new Set(live.map((m) => m.id));
    let removed = 0;
    for (const id of store.sessionIds()) {
      if (!liveIds.has(id)) {
        store.deleteSession(id);
        removed++;
      }
    }

    const stale = live.filter((m) => store.sessionUpdated(m.id) !== m.updated).sort((a, b) => b.updated - a.updated);
    const batch = stale.slice(0, options.limit ?? Number.POSITIVE_INFINITY);
    const failed: Array<{ id: string; error: string }> = [];
    let indexed = 0;
    for (const meta of batch) {
      try {
        await this.#indexOne(source, meta);
        indexed++;
      } catch (error) {
        failed.push({ id: meta.id, error: error instanceof Error ? error.message : String(error) });
      }
    }
    return {
      scanned: live.length,
      indexed,
      unchanged: live.length - stale.length,
      removed,
      remaining: stale.length - batch.length,
      rebuilt,
      failed,
    };
  }

  async #indexOne(source: SessionSource, meta: SessionMeta): Promise<void> {
    const transcript = buildTranscript(meta, await source.messages(meta.id));
    const texts = chunkText(transcript, this.#config.chunkChars, this.#config.chunkOverlap);
    const chunks: Array<{ text: string; vector: Float32Array }> = [];
    for (let i = 0; i < texts.length; i += BATCH) {
      const slice = texts.slice(i, i + BATCH);
      const vectors = await this.#embedder.embedDocuments(slice);
      slice.forEach((text, j) => chunks.push({ text, vector: vectors[j]! }));
    }
    this.#store.replaceSession(
      { id: meta.id, title: meta.title, directory: meta.directory, projectID: meta.projectID, updated: meta.updated },
      chunks,
    );
  }
}
