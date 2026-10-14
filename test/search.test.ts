import { test } from "node:test";
import assert from "node:assert/strict";
import { loadConfig, type Config } from "../src/config.ts";
import { Indexer } from "../src/indexer.ts";
import { hybridSearch } from "../src/search.ts";
import { Store } from "../src/store.ts";
import { FakeEmbedder, FakeReranker, FakeSource, session } from "./helpers.ts";

const cfg = (over: Partial<Config> = {}): Config => ({ ...loadConfig({}), threshold: 0.5, ...over });

function setup(sessions = [session("s1", "Car trouble", ["my car repair notes", "check the brakes"])], over: Partial<Config> = {}) {
  const config = cfg(over);
  const store = new Store(":memory:");
  const embedder = new FakeEmbedder();
  const source = new FakeSource(sessions);
  const indexer = new Indexer({ source: async () => source, store, embedder, config });
  return { config, store, embedder, source, indexer };
}

test("indexer: indexes new sessions, skips unchanged, reindexes updated, removes deleted", async () => {
  const { indexer, source, store } = setup([session("a", "A", ["alpha"]), session("b", "B", ["beta"])]);
  let r = await indexer.run();
  assert.deepEqual([r.scanned, r.indexed, r.unchanged, r.removed], [2, 2, 0, 0]);

  r = await indexer.run();
  assert.deepEqual([r.indexed, r.unchanged], [0, 2]);

  source.sessions[0]!.meta = { ...source.sessions[0]!.meta, updated: 2000 };
  r = await indexer.run();
  assert.equal(r.indexed, 1);

  source.sessions.pop();
  r = await indexer.run();
  assert.equal(r.removed, 1);
  assert.deepEqual(store.sessionIds(), ["a"]);
});

test("indexer: honours archived and subagent filters", async () => {
  const sessions = [
    session("top", "Top", ["x"]),
    session("old", "Old", ["x"], { archived: true }),
    session("sub", "Sub", ["x"], { subagent: true }),
  ];
  const a = setup(sessions);
  await a.indexer.run();
  assert.deepEqual(a.store.sessionIds(), ["top"]);

  const b = setup(sessions, { includeArchived: true, includeSubagents: true });
  await b.indexer.run();
  assert.deepEqual(b.store.sessionIds().sort(), ["old", "sub", "top"]);
});

test("indexer: limit indexes newest first and reports the remainder", async () => {
  const sessions = [
    session("old", "Old", ["x"], { updated: 1 }),
    session("new", "New", ["x"], { updated: 3 }),
    session("mid", "Mid", ["x"], { updated: 2 }),
  ];
  const { indexer, store } = setup(sessions);
  const r = await indexer.run({ limit: 2 });
  assert.deepEqual([r.indexed, r.remaining], [2, 1]);
  assert.deepEqual(store.sessionIds().sort(), ["mid", "new"]);
  const done = await indexer.run();
  assert.deepEqual([done.indexed, done.remaining], [1, 0]);
});

test("indexer: one failing session does not stop the rest and is retried later", async () => {
  const bad = session("bad", "Bad", ["x"]);
  bad.fail = true;
  const { indexer, store } = setup([bad, session("ok", "Ok", ["fine"])]);
  let r = await indexer.run();
  assert.equal(r.indexed, 1);
  assert.deepEqual(r.failed, [{ id: "bad", error: "boom" }]);
  bad.fail = false;
  r = await indexer.run();
  assert.equal(r.indexed, 1);
  assert.deepEqual(store.sessionIds().sort(), ["bad", "ok"]);
});

test("indexer: a changed embedder signature or force rebuilds everything", async () => {
  const { indexer, embedder, store } = setup([session("a", "A", ["alpha"])]);
  await indexer.run();
  const before = embedder.documentCalls;
  assert.equal((await indexer.run()).rebuilt, false);
  assert.equal(embedder.documentCalls, before);

  embedder.signature = "fake|v2";
  const r = await indexer.run();
  assert.equal(r.rebuilt, true);
  assert.equal(r.indexed, 1);
  assert.equal(store.getMeta("embedder"), "fake|v2");

  assert.equal((await indexer.run({ force: true })).rebuilt, true);
});

test("indexer: concurrent runs share one pass", async () => {
  const { indexer, source } = setup([session("a", "A", ["alpha"])]);
  const [x, y] = await Promise.all([indexer.run(), indexer.run()]);
  assert.equal(x, y);
  assert.equal(source.messageCalls.length, 1);
});

test("indexer: a session with no messages is recorded and not retried", async () => {
  const { indexer, source } = setup([session("empty", "Empty", [])]);
  await indexer.run();
  await indexer.run();
  assert.equal(source.messageCalls.length, 1);
});

test("search: finds a paraphrase semantically and an identifier by keyword", async () => {
  const { indexer, store, embedder, config } = setup([
    session("car", "Car trouble", ["my car repair notes"]),
    session("ticket", "Deploy", ["rolled out GAT-1234 to cert"]),
    session("misc", "Misc", ["gardening tomatoes and basil"]),
  ]);
  await indexer.run();

  const sem = await hybridSearch({ store, embedder, config }, "automobile fix", 5);
  assert.equal(sem[0]!.sessionId, "car");
  assert.ok(sem[0]!.vectorScore! > 0.6);

  const kw = await hybridSearch({ store, embedder, config }, "GAT-1234", 5);
  assert.equal(kw[0]!.sessionId, "ticket");
  assert.equal(kw[0]!.keywordMatch, true);
});

test("search: unrelated queries return nothing and topK is respected", async () => {
  const { indexer, store, embedder, config } = setup([
    session("a", "A", ["car repair one"]),
    session("b", "B", ["car repair two"]),
    session("c", "C", ["car repair three"]),
  ]);
  await indexer.run();
  assert.deepEqual(await hybridSearch({ store, embedder, config }, "quantum chromodynamics", 5), []);
  assert.equal((await hybridSearch({ store, embedder, config }, "car repair", 2)).length, 2);
});

test("search: results are collapsed to the best chunk per session", async () => {
  const long = Array.from({ length: 60 }, () => "car repair brakes").join(" ");
  const { indexer, store, embedder, config } = setup([session("one", "One", [long])], { chunkChars: 200, chunkOverlap: 20 });
  await indexer.run();
  assert.ok(store.counts().chunks > 1);
  const hits = await hybridSearch({ store, embedder, config }, "car repair", 10);
  assert.equal(hits.length, 1);
});

test("search: a session matched by both signals outranks a keyword-only one", async () => {
  const { indexer, store, embedder, config } = setup([
    session("both", "Both", ["car repair log"]),
    session("kw", "Kw", ["stray mention of repair among many unrelated words about gardens oceans planets music"]),
  ]);
  await indexer.run();
  const hits = await hybridSearch({ store, embedder, config }, "car repair", 5);
  assert.equal(hits[0]!.sessionId, "both");
});

test("search: reranker reorders candidates and rerankMinScore drops weak ones", async () => {
  const sessions = [
    session("plain", "Plain", ["car repair standard guide"]),
    session("pref", "Pref", ["car repair PREFERRED guide with extra words"]),
    session("junk", "Junk", ["car repair JUNK guide"]),
  ];
  const reranker = new FakeReranker();
  const a = setup(sessions);
  await a.indexer.run();
  const without = await hybridSearch({ store: a.store, embedder: a.embedder, config: a.config }, "car repair guide", 5);
  assert.equal(without.every((h) => !h.reranked), true);

  const withRerank = await hybridSearch({ store: a.store, embedder: a.embedder, config: a.config, reranker }, "car repair guide", 5);
  assert.equal(withRerank[0]!.sessionId, "pref");
  assert.equal(withRerank.at(-1)!.sessionId, "junk");
  assert.ok(withRerank.every((h) => h.reranked));

  const strict = setup(sessions, { rerankMinScore: -1 });
  await strict.indexer.run();
  const cut = await hybridSearch({ store: strict.store, embedder: strict.embedder, config: strict.config, reranker }, "car repair guide", 5);
  assert.deepEqual(cut.map((h) => h.sessionId).sort(), ["plain", "pref"]);
});

test("search: reranker only sees the top rerankTop candidates", async () => {
  const sessions = Array.from({ length: 6 }, (_, i) => session(`s${i}`, `S${i}`, [`car repair number ${i}`]));
  const reranker = new FakeReranker();
  const { indexer, store, embedder, config } = setup(sessions, { rerankTop: 3 });
  await indexer.run();
  await hybridSearch({ store, embedder, config, reranker }, "car repair", 10);
  assert.equal(reranker.calls[0]!.length, 3);
});
