export interface SessionMeta {
  readonly id: string;
  readonly title: string;
  readonly directory?: string;
  readonly projectID?: string;
  readonly updated: number;
  readonly archived: boolean;
  readonly subagent: boolean;
}

export interface Message {
  readonly role: "user" | "assistant";
  readonly text: string;
  readonly time: number;
}

export interface SessionSource {
  listSessions(): AsyncIterable<SessionMeta>;
  messages(sessionId: string): Promise<Message[]>;
}

export interface Embedder {
  /** Identifies everything that changes the vectors: model, pooling, prefixes. A change forces a re-index. */
  readonly signature: string;
  embedDocuments(texts: readonly string[]): Promise<Float32Array[]>;
  embedQuery(text: string): Promise<Float32Array>;
}

export interface Reranker {
  readonly id: string;
  score(query: string, passages: readonly string[]): Promise<number[]>;
}

export interface Answerer {
  readonly id: string;
  answer(query: string, passages: readonly string[]): Promise<string>;
}
