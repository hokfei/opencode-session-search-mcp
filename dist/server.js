import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { z } from "zod";
import { hybridSearch } from "./search.js";
const text = (t) => ({ content: [{ type: "text", text: t }] });
const failure = (error) => ({
    content: [{ type: "text", text: error instanceof Error ? error.message : String(error) }],
    isError: true,
});
const date = (ms) => new Date(ms).toISOString().slice(0, 10);
export function formatReport(r) {
    const lines = [
        `Indexed ${r.indexed} session(s); ${r.unchanged} unchanged, ${r.removed} removed, ${r.scanned} scanned.`,
    ];
    if (r.rebuilt)
        lines.push("The index was rebuilt from scratch (forced, or the embedding settings changed).");
    if (r.remaining > 0)
        lines.push(`${r.remaining} session(s) still waiting; call index_sessions again to continue.`);
    for (const f of r.failed)
        lines.push(`Failed ${f.id}: ${f.error}`);
    return lines.join("\n");
}
export function formatHits(query, hits) {
    if (!hits.length)
        return `No sessions matched "${query}".`;
    return hits
        .map((h, i) => {
        const how = [h.vectorScore !== undefined ? `semantic ${h.vectorScore.toFixed(2)}` : "", h.keywordMatch ? "keyword" : ""]
            .filter(Boolean)
            .join(" + ");
        const where = h.directory ? ` · ${h.directory}` : "";
        const snippet = h.snippet.replace(/\s+/g, " ");
        return `${i + 1}. ${h.title || "untitled"} (${h.sessionId}) · ${date(h.updated)}${where}\n   match: ${how || "n/a"}${h.reranked ? ` · rerank ${h.score.toFixed(2)}` : ""}\n   ${snippet}`;
    })
        .join("\n\n");
}
export function createServer(deps) {
    const { config, store, embedder, indexer, connect, reranker, answerer } = deps;
    const server = new McpServer({ name: "opencode-session-search", version: "0.1.0" });
    server.registerTool("search_sessions", {
        description: "Search past OpenCode conversations by meaning and by exact keywords (ticket ids, error codes, file names). New sessions are indexed automatically.",
        inputSchema: {
            query: z.string().min(1).describe("What to look for."),
            limit: z.number().int().min(1).max(config.maxTopK).optional().describe("Max sessions to return."),
            answer: z.boolean().optional().describe("Write a short answer from the matches (needs an answer model)."),
        },
    }, async ({ query, limit, answer }) => {
        try {
            let note = "";
            if (config.autoIndex) {
                const report = await indexer.run({ limit: config.autoIndexLimit });
                if (report.remaining > 0)
                    note = `\n\n(${report.remaining} session(s) are not indexed yet; call index_sessions to finish.)`;
                if (report.failed.length)
                    note += `\n\n(${report.failed.length} session(s) failed to index; see health_check or index_sessions.)`;
            }
            const hits = await hybridSearch({ store, embedder, reranker, config }, query, limit ?? config.topK);
            let out = formatHits(query, hits);
            if (answer !== false && answerer && hits.length) {
                const written = await answerer.answer(query, hits.slice(0, 3).map((h) => h.snippet));
                if (written)
                    out = `${written}\n\n---\n${out}`;
            }
            return text(out + note);
        }
        catch (error) {
            return failure(error);
        }
    });
    server.registerTool("get_session", {
        description: "Read one session's transcript (user and assistant text).",
        inputSchema: {
            sessionId: z.string().min(1),
            maxChars: z.number().int().min(500).max(200_000).optional().describe("Truncate the transcript (default 20000)."),
        },
    }, async ({ sessionId, maxChars }) => {
        try {
            const messages = await (await connect()).messages(sessionId);
            if (!messages.length)
                return text(`Session ${sessionId} has no readable messages.`);
            const full = messages.map((m) => `${m.role}: ${m.text}`).join("\n\n");
            const cap = maxChars ?? 20_000;
            return text(full.length > cap ? `${full.slice(0, cap)}\n\n[truncated: ${full.length - cap} more characters]` : full);
        }
        catch (error) {
            return failure(error);
        }
    });
    server.registerTool("list_sessions", {
        description: "List recently updated indexed sessions.",
        inputSchema: { limit: z.number().int().min(1).max(100).optional() },
    }, async ({ limit }) => {
        try {
            const rows = store.listSessions(limit ?? 20);
            if (!rows.length)
                return text("No sessions are indexed yet. Call index_sessions.");
            return text(rows.map((r) => `${date(r.updated)}  ${r.title || "untitled"}  (${r.id})`).join("\n"));
        }
        catch (error) {
            return failure(error);
        }
    });
    server.registerTool("index_sessions", {
        description: "Index new or changed sessions now. Rarely needed, because search does it automatically.",
        inputSchema: { limit: z.number().int().min(1).optional().describe("Max sessions to index in this call.") },
    }, async ({ limit }) => {
        try {
            return text(formatReport(await indexer.run({ limit })));
        }
        catch (error) {
            return failure(error);
        }
    });
    server.registerTool("rebuild_index", {
        description: "Wipe the index and re-embed every session. Slow; use after changing the embedding model.",
        inputSchema: { confirm: z.boolean().describe("Must be true.") },
    }, async ({ confirm }) => {
        if (!confirm)
            return failure(new Error("Set confirm=true to rebuild the index."));
        try {
            return text(formatReport(await indexer.run({ force: true })));
        }
        catch (error) {
            return failure(error);
        }
    });
    server.registerTool("health_check", { description: "Show index size, active models and whether OpenCode is reachable." }, async () => {
        const counts = store.counts();
        let opencode = "reachable";
        try {
            await connect();
        }
        catch (error) {
            opencode = error instanceof Error ? error.message : String(error);
        }
        return text([
            `Index: ${counts.sessions} session(s), ${counts.chunks} chunk(s) at ${config.dbPath}`,
            `Embedding model: ${config.embedModel} (${config.embedPooling} pooling, ${config.embedDtype})`,
            `Reranker: ${reranker ? reranker.id : "off"}`,
            `Answer model: ${answerer ? answerer.id : "off"}`,
            `Similarity threshold: ${config.threshold}`,
            `OpenCode: ${opencode}`,
        ].join("\n"));
    });
    return server;
}
