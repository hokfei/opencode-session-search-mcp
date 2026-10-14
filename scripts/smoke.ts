import { loadConfig } from "../src/config.ts";
import { Indexer } from "../src/indexer.ts";
import { TransformersEmbedder, TransformersReranker } from "../src/models.ts";
import { hybridSearch } from "../src/search.ts";
import { Store } from "../src/store.ts";
import { FakeSource, session } from "../test/helpers.ts";

// Manual check against the real models: node scripts/smoke.ts [rerank]
const config = loadConfig({});
const store = new Store(":memory:");
const embedder = new TransformersEmbedder(config);
const reranker = process.argv.includes("rerank") ? new TransformersReranker("Xenova/bge-reranker-base") : undefined;
const source = new FakeSource([
  session("pr", "Python repo CI", ["The ruff format check failed on opencode.py in the pull request.", "I reformatted two spots and CI went green."]),
  session("gat", "Cert deployment", ["Deployed GAT-2025 certificate to the esweb servers in Brampton.", "Done, verified on both nodes."]),
  session("cook", "Dinner", ["How long should I roast a chicken?", "About 75 minutes at 200C."]),
  session("gpu", "Hardware", ["My laptop has 24GB RAM and a GTX 1650 with 4GB VRAM.", "That fits small quantized models."]),
]);
const indexer = new Indexer({ source: async () => source, store, embedder, config });
console.log(JSON.stringify(await indexer.run()));

for (const q of ["linter complaint on my pull request", "GAT-2025", "how much video memory do I have", "stock market crash history"]) {
  const t = Date.now();
  const hits = await hybridSearch({ store, embedder, reranker, config }, q, 3);
  console.log(`\n${q}  (${Date.now() - t} ms)`);
  for (const h of hits) console.log(`  ${h.sessionId} score=${h.score.toFixed(3)} vec=${h.vectorScore?.toFixed(3) ?? "-"} kw=${h.keywordMatch}`);
  if (!hits.length) console.log("  (none)");
}
