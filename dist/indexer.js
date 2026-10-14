import { buildTranscript, chunkText } from "./chunking.js";
const BATCH = 16;
export class Indexer {
    #source;
    #store;
    #embedder;
    #config;
    #running;
    constructor(deps) {
        this.#source = deps.source;
        this.#store = deps.store;
        this.#embedder = deps.embedder;
        this.#config = deps.config;
    }
    /** Concurrent callers share one pass instead of racing each other. */
    run(options = {}) {
        this.#running ??= this.#run(options).finally(() => {
            this.#running = undefined;
        });
        return this.#running;
    }
    async #run(options) {
        const store = this.#store;
        const signature = this.#embedder.signature;
        const stored = store.getMeta("embedder");
        const rebuilt = Boolean(options.force) || (stored !== undefined && stored !== signature);
        if (rebuilt)
            store.clear();
        store.setMeta("embedder", signature);
        const source = await this.#source();
        const live = [];
        for await (const meta of source.listSessions()) {
            if (meta.archived && !this.#config.includeArchived)
                continue;
            if (meta.subagent && !this.#config.includeSubagents)
                continue;
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
        const failed = [];
        let indexed = 0;
        for (const meta of batch) {
            try {
                await this.#indexOne(source, meta);
                indexed++;
            }
            catch (error) {
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
    async #indexOne(source, meta) {
        const transcript = buildTranscript(meta, await source.messages(meta.id));
        const texts = chunkText(transcript, this.#config.chunkChars, this.#config.chunkOverlap);
        const chunks = [];
        for (let i = 0; i < texts.length; i += BATCH) {
            const slice = texts.slice(i, i + BATCH);
            const vectors = await this.#embedder.embedDocuments(slice);
            slice.forEach((text, j) => chunks.push({ text, vector: vectors[j] }));
        }
        this.#store.replaceSession({ id: meta.id, title: meta.title, directory: meta.directory, projectID: meta.projectID, updated: meta.updated }, chunks);
    }
}
