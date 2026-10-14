import { OpenCode } from "@opencode/client";
import { Service } from "@opencode/client/service";
const PAGE = 100;
export function toMessage(m) {
    const time = m.time?.created ?? 0;
    if (m.type === "user" && m.text)
        return { role: "user", text: m.text, time };
    if (m.type === "assistant") {
        const text = (m.content ?? [])
            .filter((part) => part.type === "text" && part.text)
            .map((part) => part.text)
            .join("\n");
        if (text)
            return { role: "assistant", text, time };
    }
    return undefined;
}
export class OpenCodeSource {
    #client;
    constructor(client) {
        this.#client = client;
    }
    async *listSessions() {
        let cursor;
        do {
            const page = await this.#client.session.list({ limit: PAGE, order: "desc", cursor });
            for (const s of page.data) {
                yield {
                    id: s.id,
                    title: s.title ?? "",
                    directory: s.subpath,
                    projectID: s.projectID,
                    updated: s.time.updated,
                    archived: s.time.archived !== undefined,
                    subagent: s.parentID !== undefined,
                };
            }
            cursor = page.cursor.next ?? undefined;
        } while (cursor);
    }
    async messages(sessionId) {
        const out = [];
        let cursor;
        do {
            const page = await this.#client.message.list({ sessionID: sessionId, limit: 200, order: "asc", cursor });
            for (const m of page.data) {
                const message = toMessage(m);
                if (message)
                    out.push(message);
            }
            cursor = page.cursor.next ?? undefined;
        } while (cursor);
        return out;
    }
}
/** Finds the running OpenCode service. It never starts one: this server is launched by OpenCode itself. */
export async function connectOpenCode() {
    const endpoint = await Service.discover();
    if (!endpoint)
        throw new Error("No running OpenCode service found. Start OpenCode first, then retry.");
    const client = OpenCode.make({ baseUrl: endpoint.url, headers: Service.headers(endpoint) });
    return new OpenCodeSource(client);
}
