#!/usr/bin/env node
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { loadConfig } from "./config.ts";
import { Indexer } from "./indexer.ts";
import { TransformersAnswerer, TransformersEmbedder, TransformersReranker } from "./models.ts";
import { connectOpenCode } from "./opencode.ts";
import { createServer } from "./server.ts";
import { Store } from "./store.ts";

const config = loadConfig();
const store = new Store(config.dbPath);
const embedder = new TransformersEmbedder(config);
const reranker = config.rerankModel ? new TransformersReranker(config.rerankModel) : undefined;
const answerer = config.answerModel ? new TransformersAnswerer(config.answerModel) : undefined;
const indexer = new Indexer({ source: connectOpenCode, store, embedder, config });

const server = createServer({ config, store, embedder, indexer, connect: connectOpenCode, reranker, answerer });
await server.connect(new StdioServerTransport());
