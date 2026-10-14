import { OpenCode } from "@opencode/client";
import { Service } from "@opencode/client/service";
import type { Message, SessionMeta, SessionSource } from "./types.ts";

const PAGE = 100;

interface SessionLike {
  readonly id: string;
  readonly parentID?: string;
  readonly projectID: string;
  readonly title?: string;
  readonly subpath?: string;
  readonly time: { readonly updated: number; readonly archived?: number };
}

interface MessageLike {
  readonly type: string;
  readonly time?: { readonly created: number };
  readonly text?: string;
  readonly content?: ReadonlyArray<{ readonly type: string; readonly text?: string }>;
}

interface Page<T> {
  readonly data: readonly T[];
  readonly cursor: { readonly next?: string | null };
}

/** The slice of the OpenCode client this server needs. Kept narrow so tests can fake it. */
export interface ClientLike {
  readonly session: {
    list(input: { limit: number; order?: "desc"; cursor?: string }): Promise<Page<SessionLike>>;
  };
  readonly message: {
    list(input: { sessionID: string; limit: number; order?: "asc"; cursor?: string }): Promise<Page<MessageLike>>;
  };
}

export function toMessage(m: MessageLike): Message | undefined {
  const time = m.time?.created ?? 0;
  if (m.type === "user" && m.text) return { role: "user", text: m.text, time };
  if (m.type === "assistant") {
    const text = (m.content ?? [])
      .filter((part) => part.type === "text" && part.text)
      .map((part) => part.text)
      .join("\n");
    if (text) return { role: "assistant", text, time };
  }
  return undefined;
}

export class OpenCodeSource implements SessionSource {
  readonly #client: ClientLike;

  constructor(client: ClientLike) {
    this.#client = client;
  }

  async *listSessions(): AsyncIterable<SessionMeta> {
    let cursor: string | undefined;
    do {
      // The API rejects `order` together with `cursor`; the cursor carries the ordering.
      const page = await this.#client.session.list(
        cursor ? { limit: PAGE, cursor } : { limit: PAGE, order: "desc" },
      );
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

  async messages(sessionId: string): Promise<Message[]> {
    const out: Message[] = [];
    let cursor: string | undefined;
    do {
      const page = await this.#client.message.list(
        cursor ? { sessionID: sessionId, limit: 200, cursor } : { sessionID: sessionId, limit: 200, order: "asc" },
      );
      for (const m of page.data) {
        const message = toMessage(m);
        if (message) out.push(message);
      }
      cursor = page.cursor.next ?? undefined;
    } while (cursor);
    return out;
  }
}

/** Finds the running OpenCode service. It never starts one: this server is launched by OpenCode itself. */
export async function connectOpenCode(): Promise<SessionSource> {
  const endpoint = await Service.discover();
  if (!endpoint) throw new Error("No running OpenCode service found. Start OpenCode first, then retry.");
  const client = OpenCode.make({ baseUrl: endpoint.url, headers: Service.headers(endpoint) });
  return new OpenCodeSource(client as unknown as ClientLike);
}
