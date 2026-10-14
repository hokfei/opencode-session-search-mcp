import { mkdirSync } from "node:fs";
import { dirname } from "node:path";
import { DatabaseSync } from "node:sqlite";
const STOPWORDS = new Set("a an and are as at be but by do does did for from had has have how i if in into is it its me my of on or our so than that the their them then there these they this to was we were what when where which who why will with you your".split(" "));
/**
 * Turns free text into an FTS5 query. Terms are OR-ed so bm25 can rank partial matches.
 * Identifiers such as GAT-1234 or session_search.py also become a phrase, so the whole
 * identifier scores higher than its parts.
 */
export function buildFtsQuery(query) {
    const terms = new Set();
    const phrases = new Set();
    for (const raw of query.split(/\s+/)) {
        const words = raw.toLowerCase().match(/[\p{L}\p{N}]+/gu) ?? [];
        const useful = words.filter((w) => !STOPWORDS.has(w) && w.length > 1);
        for (const w of useful)
            terms.add(w);
        if (words.length > 1 && /[-_./]/.test(raw) && words.every((w) => w.length > 0)) {
            phrases.add(words.join(" "));
        }
    }
    const parts = [...[...phrases].map((p) => `"${p}"`), ...[...terms].map((t) => `"${t}"`)];
    return parts.length ? parts.join(" OR ") : undefined;
}
export class Store {
    #db;
    #matrix;
    constructor(path) {
        if (path !== ":memory:")
            mkdirSync(dirname(path), { recursive: true });
        this.#db = new DatabaseSync(path);
        this.#db.exec(`
      PRAGMA journal_mode = WAL;
      PRAGMA foreign_keys = ON;
      CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
      CREATE TABLE IF NOT EXISTS sessions (
        id TEXT PRIMARY KEY, title TEXT NOT NULL, directory TEXT, project_id TEXT,
        updated INTEGER NOT NULL, indexed_at INTEGER NOT NULL
      );
      CREATE TABLE IF NOT EXISTS chunks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
        idx INTEGER NOT NULL, text TEXT NOT NULL, vector BLOB NOT NULL
      );
      CREATE INDEX IF NOT EXISTS chunks_session ON chunks(session_id);
      CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(text, tokenize = 'porter unicode61');
    `);
    }
    getMeta(key) {
        const row = this.#db.prepare("SELECT value FROM meta WHERE key = ?").get(key);
        return row?.value;
    }
    setMeta(key, value) {
        this.#db
            .prepare("INSERT INTO meta(key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value")
            .run(key, value);
    }
    sessionUpdated(id) {
        const row = this.#db.prepare("SELECT updated FROM sessions WHERE id = ?").get(id);
        return row?.updated;
    }
    sessionIds() {
        return this.#db.prepare("SELECT id FROM sessions").all().map((r) => r.id);
    }
    replaceSession(session, chunks, now = Date.now()) {
        this.#transaction(() => {
            this.#removeChunks(session.id);
            this.#db
                .prepare(`INSERT INTO sessions(id, title, directory, project_id, updated, indexed_at) VALUES (?, ?, ?, ?, ?, ?)
           ON CONFLICT(id) DO UPDATE SET title = excluded.title, directory = excluded.directory,
             project_id = excluded.project_id, updated = excluded.updated, indexed_at = excluded.indexed_at`)
                .run(session.id, session.title, session.directory ?? null, session.projectID ?? null, session.updated, now);
            const insert = this.#db.prepare("INSERT INTO chunks(session_id, idx, text, vector) VALUES (?, ?, ?, ?)");
            const fts = this.#db.prepare("INSERT INTO chunks_fts(rowid, text) VALUES (?, ?)");
            chunks.forEach((c, idx) => {
                const bytes = new Uint8Array(c.vector.buffer, c.vector.byteOffset, c.vector.byteLength);
                const r = insert.run(session.id, idx, c.text, bytes);
                fts.run(Number(r.lastInsertRowid), c.text);
            });
        });
        this.#matrix = undefined;
    }
    deleteSession(id) {
        this.#transaction(() => {
            this.#removeChunks(id);
            this.#db.prepare("DELETE FROM sessions WHERE id = ?").run(id);
        });
        this.#matrix = undefined;
    }
    clear() {
        this.#transaction(() => {
            this.#db.exec("DELETE FROM chunks_fts; DELETE FROM chunks; DELETE FROM sessions;");
        });
        this.#matrix = undefined;
    }
    /** Exact cosine search over unit vectors. Scores every chunk, which is cheap well past 100k chunks. */
    vectorSearch(query, limit, minScore) {
        const m = this.#loadMatrix();
        if (!m || m.dim !== query.length)
            return [];
        const scored = [];
        for (let i = 0; i < m.ids.length; i++) {
            const base = i * m.dim;
            let dot = 0;
            for (let d = 0; d < m.dim; d++)
                dot += m.data[base + d] * query[d];
            if (dot >= minScore)
                scored.push({ chunkId: m.ids[i], sessionId: m.sessions[i], score: dot });
        }
        scored.sort((a, b) => b.score - a.score);
        return scored.slice(0, limit);
    }
    keywordSearch(query, limit) {
        const match = buildFtsQuery(query);
        if (!match)
            return [];
        const rows = this.#db
            .prepare(`SELECT c.id AS chunkId, c.session_id AS sessionId
         FROM chunks_fts JOIN chunks c ON c.id = chunks_fts.rowid
         WHERE chunks_fts MATCH ? ORDER BY chunks_fts.rank LIMIT ?`)
            .all(match, limit);
        return rows.map((r) => ({ chunkId: Number(r.chunkId), sessionId: r.sessionId }));
    }
    chunks(ids) {
        const out = new Map();
        const stmt = this.#db.prepare("SELECT id, session_id AS sessionId, text FROM chunks WHERE id = ?");
        for (const id of ids) {
            const r = stmt.get(id);
            if (r)
                out.set(id, { chunkId: r.id, sessionId: r.sessionId, text: r.text });
        }
        return out;
    }
    sessionInfo(id) {
        const r = this.#db
            .prepare("SELECT id, title, directory, project_id AS projectID, updated, indexed_at AS indexedAt FROM sessions WHERE id = ?")
            .get(id);
        return r && { ...r, directory: r.directory ?? undefined, projectID: r.projectID ?? undefined };
    }
    listSessions(limit) {
        const rows = this.#db
            .prepare("SELECT id, title, directory, project_id AS projectID, updated FROM sessions ORDER BY updated DESC LIMIT ?")
            .all(limit);
        return rows.map((r) => ({ ...r, directory: r.directory ?? undefined, projectID: r.projectID ?? undefined }));
    }
    counts() {
        const s = this.#db.prepare("SELECT count(*) AS n FROM sessions").get();
        const c = this.#db.prepare("SELECT count(*) AS n FROM chunks").get();
        return { sessions: Number(s.n), chunks: Number(c.n) };
    }
    close() {
        this.#db.close();
    }
    #removeChunks(sessionId) {
        this.#db
            .prepare("DELETE FROM chunks_fts WHERE rowid IN (SELECT id FROM chunks WHERE session_id = ?)")
            .run(sessionId);
        this.#db.prepare("DELETE FROM chunks WHERE session_id = ?").run(sessionId);
    }
    #transaction(fn) {
        this.#db.exec("BEGIN");
        try {
            fn();
            this.#db.exec("COMMIT");
        }
        catch (error) {
            this.#db.exec("ROLLBACK");
            throw error;
        }
    }
    #loadMatrix() {
        if (this.#matrix)
            return this.#matrix.ids.length ? this.#matrix : undefined;
        const rows = this.#db
            .prepare("SELECT id, session_id AS sessionId, vector FROM chunks ORDER BY id")
            .all();
        const dim = rows.length ? rows[0].vector.byteLength / 4 : 0;
        const data = new Float32Array(rows.length * dim);
        const ids = new Int32Array(rows.length);
        const sessions = [];
        rows.forEach((r, i) => {
            ids[i] = Number(r.id);
            sessions.push(r.sessionId);
            data.set(new Float32Array(r.vector.buffer.slice(r.vector.byteOffset, r.vector.byteOffset + r.vector.byteLength)), i * dim);
        });
        this.#matrix = { ids, sessions, data, dim };
        return rows.length ? this.#matrix : undefined;
    }
}
