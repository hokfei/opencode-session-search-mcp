import { test } from "node:test";
import assert from "node:assert/strict";
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { InMemoryTransport } from "@modelcontextprotocol/sdk/inMemory.js";
import { loadConfig } from "../src/config.ts";
import { Indexer } from "../src/indexer.ts";
import { OpenCodeSource, toMessage, type ClientLike } from "../src/opencode.ts";
import { createServer } from "../src/server.ts";
import { Store } from "../src/store.ts";
import { FakeAnswerer, FakeEmbedder, FakeReranker, FakeSource, session } from "./helpers.ts";

async function connected(opts: { answerer?: boolean; reranker?: boolean; offline?: boolean; autoIndexLimit?: number } = {}) {
  const config = { ...loadConfig({}), threshold: 0.5, autoIndexLimit: opts.autoIndexLimit ?? 200 };
  const store = new Store(":memory:");
  const embedder = new FakeEmbedder();
  const source = new FakeSource([
    session("car", "Car trouble", ["my car repair notes"]),
    session("ticket", "Deploy", ["rolled out GAT-1234 to cert"]),
  ]);
  const connect = async () => {
    if (opts.offline) throw new Error("No running OpenCode service found. Start OpenCode first, then retry.");
    return source;
  };
  const indexer = new Indexer({ source: connect, store, embedder, config });
  const server = createServer({
    config,
    store,
    embedder,
    indexer,
    connect,
    reranker: opts.reranker ? new FakeReranker() : undefined,
    answerer: opts.answerer ? new FakeAnswerer() : undefined,
  });
  const [a, b] = InMemoryTransport.createLinkedPair();
  const client = new Client({ name: "t", version: "0" });
  await Promise.all([server.connect(a), client.connect(b)]);
  const call = async (name: string, args: Record<string, unknown> = {}) => {
    const r = (await client.callTool({ name, arguments: args })) as { content: Array<{ text: string }>; isError?: boolean };
    return { text: r.content[0]!.text, isError: Boolean(r.isError) };
  };
  return { client, call, store, source };
}

test("server lists the six tools", async () => {
  const { client } = await connected();
  const names = (await client.listTools()).tools.map((t) => t.name).sort();
  assert.deepEqual(names, ["get_session", "health_check", "index_sessions", "list_sessions", "rebuild_index", "search_sessions"]);
});

test("search_sessions auto-indexes, then finds by meaning and by keyword", async () => {
  const { call, store } = await connected();
  assert.equal(store.counts().sessions, 0);
  const sem = await call("search_sessions", { query: "automobile fix" });
  assert.equal(sem.isError, false);
  assert.match(sem.text, /1\. Car trouble \(car\)/);
  assert.match(sem.text, /semantic/);
  assert.equal(store.counts().sessions, 2);

  const kw = await call("search_sessions", { query: "GAT-1234" });
  assert.match(kw.text, /1\. Deploy \(ticket\)/);
  assert.match(kw.text, /keyword/);
});

test("search_sessions reports no match, and notes unfinished indexing", async () => {
  const { call } = await connected({ autoIndexLimit: 1 });
  const r = await call("search_sessions", { query: "quantum chromodynamics" });
  assert.match(r.text, /No sessions matched/);
  assert.match(r.text, /1 session\(s\) are not indexed yet/);
});

test("search_sessions prepends an answer only when an answer model exists, and answer=false skips it", async () => {
  const withAnswer = await connected({ answerer: true });
  const r = await withAnswer.call("search_sessions", { query: "car repair" });
  assert.match(r.text, /^answer\(car repair\) from 1 passage\(s\)/);
  const off = await withAnswer.call("search_sessions", { query: "car repair", answer: false });
  assert.doesNotMatch(off.text, /^answer\(/);

  const none = await connected();
  assert.doesNotMatch((await none.call("search_sessions", { query: "car repair" })).text, /answer\(/);
});

test("search_sessions with a reranker marks results as reranked", async () => {
  const { call } = await connected({ reranker: true });
  assert.match((await call("search_sessions", { query: "car repair" })).text, /rerank/);
});

test("search_sessions surfaces a missing OpenCode service as an error", async () => {
  const { call } = await connected({ offline: true });
  const r = await call("search_sessions", { query: "anything" });
  assert.equal(r.isError, true);
  assert.match(r.text, /No running OpenCode service/);
});

test("get_session returns the transcript and truncates", async () => {
  const { call, source } = await connected();
  source.sessions[0]!.messages = [{ role: "user", text: "x".repeat(900), time: 1 }];
  const full = await call("get_session", { sessionId: "car", maxChars: 500 });
  assert.match(full.text, /^user: x+/);
  assert.match(full.text, /\[truncated: \d+ more characters\]/);
  assert.equal((await call("get_session", { sessionId: "missing" })).isError, true);
});

test("index_sessions, list_sessions and rebuild_index", async () => {
  const { call } = await connected();
  assert.match((await call("list_sessions")).text, /No sessions are indexed yet/);
  assert.match((await call("index_sessions")).text, /Indexed 2 session\(s\)/);
  const list = await call("list_sessions", { limit: 5 });
  assert.match(list.text, /Car trouble {2}\(car\)/);

  const refused = await call("rebuild_index", { confirm: false });
  assert.equal(refused.isError, true);
  const done = await call("rebuild_index", { confirm: true });
  assert.match(done.text, /rebuilt from scratch/);
  assert.match(done.text, /Indexed 2 session\(s\)/);
});

test("health_check reports models, counts and connectivity", async () => {
  const ok = await connected({ reranker: true, answerer: true });
  await ok.call("index_sessions");
  const t = (await ok.call("health_check")).text;
  assert.match(t, /Index: 2 session\(s\)/);
  assert.match(t, /Reranker: fake-reranker/);
  assert.match(t, /Answer model: fake-answerer/);
  assert.match(t, /OpenCode: reachable/);

  const off = await connected({ offline: true });
  const o = await off.call("health_check");
  assert.equal(o.isError, false);
  assert.match(o.text, /OpenCode: No running OpenCode service/);
  assert.match(o.text, /Reranker: off/);
});

test("toMessage keeps user text and assistant text parts only", () => {
  assert.deepEqual(toMessage({ type: "user", text: "hi", time: { created: 5 } }), { role: "user", text: "hi", time: 5 });
  assert.deepEqual(
    toMessage({
      type: "assistant",
      content: [
        { type: "reasoning", text: "thinking" },
        { type: "text", text: "one" },
        { type: "tool" },
        { type: "text", text: "two" },
      ],
    }),
    { role: "assistant", text: "one\ntwo", time: 0 },
  );
  assert.equal(toMessage({ type: "assistant", content: [{ type: "tool" }] }), undefined);
  assert.equal(toMessage({ type: "shell", text: "ls" }), undefined);
  assert.equal(toMessage({ type: "user", text: "" }), undefined);
});

test("OpenCodeSource paginates sessions and messages and maps metadata", async () => {
  const sessionPages: Record<string, { data: any[]; cursor: { next?: string } }> = {
    "": {
      data: [
        { id: "s1", projectID: "p", title: "One", subpath: "src", time: { updated: 9 } },
        { id: "s2", projectID: "p", parentID: "s1", time: { updated: 8, archived: 7 } },
      ],
      cursor: { next: "c2" },
    },
    c2: { data: [{ id: "s3", projectID: "p", title: "Three", time: { updated: 1 } }], cursor: {} },
  };
  const messagePages: Record<string, { data: any[]; cursor: { next?: string } }> = {
    "": { data: [{ type: "user", text: "q", time: { created: 1 } }], cursor: { next: "m2" } },
    m2: { data: [{ type: "assistant", content: [{ type: "text", text: "a" }] }], cursor: {} },
  };
  const client: ClientLike = {
    session: { list: async (i) => sessionPages[i.cursor ?? ""]! },
    message: { list: async (i) => messagePages[i.cursor ?? ""]! },
  };
  const source = new OpenCodeSource(client);
  const metas = [];
  for await (const m of source.listSessions()) metas.push(m);
  assert.deepEqual(metas.map((m) => m.id), ["s1", "s2", "s3"]);
  assert.deepEqual(metas[0], { id: "s1", title: "One", directory: "src", projectID: "p", updated: 9, archived: false, subagent: false });
  assert.deepEqual([metas[1]!.archived, metas[1]!.subagent, metas[1]!.title], [true, true, ""]);
  assert.deepEqual((await source.messages("s1")).map((m) => m.text), ["q", "a"]);
});
