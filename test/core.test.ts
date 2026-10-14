import { test } from "node:test";
import assert from "node:assert/strict";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { buildTranscript, chunkText } from "../src/chunking.ts";
import { loadConfig } from "../src/config.ts";
import { Store, buildFtsQuery } from "../src/store.ts";
import { fakeVector } from "./helpers.ts";

test("chunkText returns one chunk for short text and nothing for blank text", () => {
  assert.deepEqual(chunkText("hello world", 100, 20), ["hello world"]);
  assert.deepEqual(chunkText("   \n ", 100, 20), []);
});

test("chunkText overlaps windows, covers every word and never splits a word", () => {
  const words = Array.from({ length: 400 }, (_, i) => `word${i}`);
  const chunks = chunkText(words.join(" "), 200, 50);
  assert.ok(chunks.length > 5);
  for (const c of chunks) assert.ok(c.length <= 200, `chunk too long: ${c.length}`);
  const seen = new Set(chunks.flatMap((c) => c.split(" ")));
  for (const w of words) assert.ok(seen.has(w), `missing ${w}`);
  const overlap = chunks[0]!.split(" ").filter((w) => chunks[1]!.split(" ").includes(w));
  assert.ok(overlap.length > 0, "adjacent chunks share text");
});

test("chunkText terminates on text without whitespace", () => {
  const chunks = chunkText("x".repeat(1000), 100, 30);
  assert.ok(chunks.length >= 10 && chunks.every((c) => c.length <= 100));
  assert.throws(() => chunkText("abc", 0, 0), RangeError);
});

test("buildTranscript labels roles, keeps the title and skips empty messages", () => {
  const t = buildTranscript({ title: "Fix CI" }, [
    { role: "user", text: " run ruff ", time: 1 },
    { role: "assistant", text: "   ", time: 2 },
    { role: "assistant", text: "done", time: 3 },
  ]);
  assert.equal(t, "Title: Fix CI\n\nuser: run ruff\n\nassistant: done");
});

test("buildFtsQuery drops stopwords, keeps identifiers as phrases, and is injection safe", () => {
  assert.equal(buildFtsQuery("what is the python version"), '"python" OR "version"');
  const q = buildFtsQuery("deploy GAT-1234 now")!;
  assert.ok(q.startsWith('"gat 1234"'));
  assert.ok(q.includes('"gat"') && q.includes('"1234"') && q.includes('"deploy"'));
  assert.equal(buildFtsQuery("the of a"), undefined);
  assert.equal(buildFtsQuery('" ( ) * : ^ -'), undefined);
  assert.equal(buildFtsQuery("OR NOT NEAR"), '"not" OR "near"');
  assert.equal(buildFtsQuery('foo" OR "bar'), '"foo" OR "bar"');
});

const row = (id: string, updated = 1) => ({ id, title: `Title ${id}`, updated });
const chunk = (text: string) => ({ text, vector: fakeVector(text) });

test("store: keyword search finds identifiers and ignores replaced text", () => {
  const store = new Store(":memory:");
  store.replaceSession(row("s1"), [chunk("fix for GAT-1234 deployment"), chunk("unrelated notes")]);
  assert.equal(store.keywordSearch("GAT-1234", 10).length, 1);
  store.replaceSession(row("s1", 2), [chunk("only unrelated notes now")]);
  assert.equal(store.keywordSearch("GAT-1234", 10).length, 0);
  assert.equal(store.counts().chunks, 1);
  assert.equal(store.sessionUpdated("s1"), 2);
});

test("store: vector search orders by cosine, applies the threshold and limit", () => {
  const store = new Store(":memory:");
  store.replaceSession(row("car"), [chunk("car repair")]);
  store.replaceSession(row("cat"), [chunk("cat food")]);
  const hits = store.vectorSearch(fakeVector("automobile fix"), 10, 0.5);
  assert.deepEqual(hits.map((h) => h.sessionId), ["car"]);
  assert.ok(hits[0]!.score > 0.99);
  assert.equal(store.vectorSearch(fakeVector("automobile fix"), 10, -1).length, 2);
  assert.equal(store.vectorSearch(fakeVector("automobile fix"), 1, -1).length, 1);
  assert.deepEqual(store.vectorSearch(new Float32Array(3), 10, 0), []);
});

test("store: deleteSession and clear remove vectors and keywords", () => {
  const store = new Store(":memory:");
  store.replaceSession(row("a"), [chunk("alpha text")]);
  store.replaceSession(row("b"), [chunk("beta text")]);
  store.deleteSession("a");
  assert.deepEqual(store.sessionIds(), ["b"]);
  assert.equal(store.keywordSearch("alpha", 5).length, 0);
  assert.equal(store.vectorSearch(fakeVector("alpha"), 5, 0.1).length, 0);
  store.clear();
  assert.deepEqual(store.counts(), { sessions: 0, chunks: 0 });
});

test("store persists across reopen", () => {
  const dir = mkdtempSync(join(tmpdir(), "oss-"));
  try {
    const path = join(dir, "nested", "index.db");
    const first = new Store(path);
    first.replaceSession(row("s"), [chunk("persistent car repair")]);
    first.setMeta("embedder", "x");
    first.close();
    const second = new Store(path);
    assert.equal(second.getMeta("embedder"), "x");
    assert.equal(second.vectorSearch(fakeVector("car repair"), 5, 0.3).length, 1);
    assert.equal(second.keywordSearch("persistent", 5).length, 1);
    second.close();
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test("config: defaults, presets, explicit overrides and bad values", () => {
  const d = loadConfig({});
  assert.equal(d.embedModel, "Xenova/bge-small-en-v1.5");
  assert.equal(d.embedPooling, "cls");
  assert.equal(d.threshold, 0.6);
  assert.equal(d.rerankModel, undefined);
  assert.equal(d.autoIndex, true);

  const n = loadConfig({ SESSION_SEARCH_EMBED_MODEL: "nomic-ai/nomic-embed-text-v1.5" });
  assert.equal(n.embedPooling, "mean");
  assert.equal(n.queryPrefix, "search_query: ");
  assert.equal(n.documentPrefix, "search_document: ");

  const o = loadConfig({
    SESSION_SEARCH_EMBED_MODEL: "nomic-ai/nomic-embed-text-v1.5",
    SESSION_SEARCH_QUERY_PREFIX: "",
    SESSION_SEARCH_EMBED_POOLING: "cls",
    SESSION_SEARCH_RERANK_MODEL: "Xenova/bge-reranker-base",
    SESSION_SEARCH_RERANK_MIN_SCORE: "-1.5",
    SESSION_SEARCH_AUTO_INDEX: "off",
  });
  assert.equal(o.queryPrefix, "");
  assert.equal(o.embedPooling, "cls");
  assert.equal(o.rerankModel, "Xenova/bge-reranker-base");
  assert.equal(o.rerankMinScore, -1.5);
  assert.equal(o.autoIndex, false);

  const bad = loadConfig({
    SESSION_SEARCH_SIMILARITY_THRESHOLD: "9",
    SESSION_SEARCH_DEFAULT_TOP_K: "abc",
    SESSION_SEARCH_CHUNK_CHARS: "10",
    SESSION_SEARCH_RERANK_MIN_SCORE: "nan",
    SESSION_SEARCH_EMBED_POOLING: "max",
  });
  assert.equal(bad.threshold, 0.6);
  assert.equal(bad.topK, 5);
  assert.equal(bad.chunkChars, 1200);
  assert.equal(bad.rerankMinScore, undefined);
  assert.equal(bad.embedPooling, "cls");
});
