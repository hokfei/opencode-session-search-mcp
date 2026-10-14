import type { Answerer, Embedder, Message, Reranker, SessionMeta, SessionSource } from "../src/types.ts";

const DIM = 256;
const IGNORED = new Set(["title", "user", "assistant", "untitled", "the", "a", "an", "of", "to", "is", "my"]);
const SYNONYMS: Record<string, string> = { automobile: "car", vehicle: "car", cars: "car", fix: "repair", fixing: "repair" };

function hash(word: string): number {
  let h = 2166136261;
  for (const ch of word) h = Math.imul(h ^ ch.charCodeAt(0), 16777619);
  return (h >>> 0) % DIM;
}

/** A bag-of-words embedder with a tiny synonym table, so "automobile fix" lands on "car repair". */
export function fakeVector(text: string): Float32Array {
  const v = new Float32Array(DIM);
  for (const raw of text.toLowerCase().match(/[a-z0-9]+/g) ?? []) {
    if (IGNORED.has(raw)) continue;
    v[hash(SYNONYMS[raw] ?? raw)]! += 1;
  }
  const norm = Math.hypot(...v) || 1;
  return v.map((x) => x / norm);
}

export class FakeEmbedder implements Embedder {
  signature = "fake|v1";
  documentCalls = 0;
  async embedDocuments(texts: readonly string[]): Promise<Float32Array[]> {
    this.documentCalls += texts.length;
    return texts.map(fakeVector);
  }
  async embedQuery(text: string): Promise<Float32Array> {
    return fakeVector(text);
  }
}

export interface FakeSession {
  meta: SessionMeta;
  messages: Message[];
  fail?: boolean;
}

export const session = (id: string, title: string, bodies: string[], extra: Partial<SessionMeta> = {}): FakeSession => ({
  meta: { id, title, updated: 1000, archived: false, subagent: false, ...extra },
  messages: bodies.map((text, i) => ({ role: i % 2 === 0 ? "user" : "assistant", text, time: i })),
});

export class FakeSource implements SessionSource {
  sessions: FakeSession[];
  messageCalls: string[] = [];
  constructor(sessions: FakeSession[] = []) {
    this.sessions = sessions;
  }
  async *listSessions(): AsyncIterable<SessionMeta> {
    for (const s of this.sessions) yield s.meta;
  }
  async messages(id: string): Promise<Message[]> {
    this.messageCalls.push(id);
    const s = this.sessions.find((x) => x.meta.id === id);
    if (!s) throw new Error(`no session ${id}`);
    if (s.fail) throw new Error("boom");
    return s.messages;
  }
}

export class FakeReranker implements Reranker {
  id = "fake-reranker";
  calls: string[][] = [];
  async score(_query: string, passages: readonly string[]): Promise<number[]> {
    this.calls.push([...passages]);
    return passages.map((p) => (p.includes("PREFERRED") ? 5 : p.includes("JUNK") ? -8 : 0));
  }
}

export class FakeAnswerer implements Answerer {
  id = "fake-answerer";
  async answer(query: string, passages: readonly string[]): Promise<string> {
    return `answer(${query}) from ${passages.length} passage(s)`;
  }
}
