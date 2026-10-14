import { homedir } from "node:os";
import { join } from "node:path";

export type Pooling = "cls" | "mean";

export interface Config {
  readonly dbPath: string;
  readonly embedModel: string;
  readonly embedPooling: Pooling;
  readonly embedDtype: string;
  readonly queryPrefix: string;
  readonly documentPrefix: string;
  readonly rerankModel?: string;
  readonly answerModel?: string;
  readonly threshold: number;
  readonly topK: number;
  readonly maxTopK: number;
  readonly candidates: number;
  readonly rerankTop: number;
  readonly rerankMinScore?: number;
  readonly chunkChars: number;
  readonly chunkOverlap: number;
  readonly includeArchived: boolean;
  readonly includeSubagents: boolean;
  readonly autoIndex: boolean;
  readonly autoIndexLimit: number;
}

type Env = Readonly<Record<string, string | undefined>>;

interface Preset {
  readonly pooling: Pooling;
  readonly queryPrefix?: string;
  readonly documentPrefix?: string;
}

/** Conventions for known embedding models. Explicit environment variables always win. */
export const PRESETS: Readonly<Record<string, Preset>> = {
  "Xenova/bge-small-en-v1.5": { pooling: "cls" },
  "Xenova/bge-base-en-v1.5": { pooling: "cls" },
  "Xenova/all-MiniLM-L6-v2": { pooling: "mean" },
  "nomic-ai/nomic-embed-text-v1.5": {
    pooling: "mean",
    queryPrefix: "search_query: ",
    documentPrefix: "search_document: ",
  },
};

export const DEFAULT_EMBED_MODEL = "Xenova/bge-small-en-v1.5";

function num(env: Env, name: string, fallback: number, min: number, max: number): number {
  const raw = env[`SESSION_SEARCH_${name}`];
  if (raw === undefined || raw.trim() === "") return fallback;
  const n = Number(raw);
  return Number.isFinite(n) && n >= min && n <= max ? n : fallback;
}

function optNum(env: Env, name: string, min: number, max: number): number | undefined {
  const raw = env[`SESSION_SEARCH_${name}`];
  if (raw === undefined || raw.trim() === "") return undefined;
  const n = Number(raw);
  return Number.isFinite(n) && n >= min && n <= max ? n : undefined;
}

function bool(env: Env, name: string, fallback: boolean): boolean {
  const raw = env[`SESSION_SEARCH_${name}`]?.trim().toLowerCase();
  if (raw === undefined || raw === "") return fallback;
  if (["1", "true", "yes", "on"].includes(raw)) return true;
  if (["0", "false", "no", "off"].includes(raw)) return false;
  return fallback;
}

function str(env: Env, name: string): string | undefined {
  const raw = env[`SESSION_SEARCH_${name}`]?.trim();
  return raw ? raw : undefined;
}

export function loadConfig(env: Env = process.env): Config {
  const embedModel = str(env, "EMBED_MODEL") ?? DEFAULT_EMBED_MODEL;
  const preset = PRESETS[embedModel];
  const poolingRaw = str(env, "EMBED_POOLING");
  const pooling: Pooling =
    poolingRaw === "cls" || poolingRaw === "mean" ? poolingRaw : (preset?.pooling ?? "mean");
  const chunkChars = num(env, "CHUNK_CHARS", 1200, 200, 4000);
  return {
    dbPath: str(env, "DB_PATH") ?? join(homedir(), ".opencode-session-search", "index.db"),
    embedModel,
    embedPooling: pooling,
    embedDtype: str(env, "EMBED_DTYPE") ?? "q8",
    queryPrefix: env.SESSION_SEARCH_QUERY_PREFIX ?? preset?.queryPrefix ?? "",
    documentPrefix: env.SESSION_SEARCH_DOCUMENT_PREFIX ?? preset?.documentPrefix ?? "",
    rerankModel: str(env, "RERANK_MODEL"),
    answerModel: str(env, "ANSWER_MODEL"),
    threshold: num(env, "SIMILARITY_THRESHOLD", 0.6, 0, 1),
    topK: num(env, "DEFAULT_TOP_K", 5, 1, 50),
    maxTopK: num(env, "MAX_TOP_K", 20, 1, 100),
    candidates: num(env, "CANDIDATES", 50, 5, 500),
    rerankTop: num(env, "RERANK_TOP", 20, 1, 100),
    rerankMinScore: optNum(env, "RERANK_MIN_SCORE", -100, 100),
    chunkChars,
    chunkOverlap: Math.min(num(env, "CHUNK_OVERLAP", 200, 0, 1000), Math.floor(chunkChars / 2)),
    includeArchived: bool(env, "INCLUDE_ARCHIVED", false),
    includeSubagents: bool(env, "INCLUDE_SUBAGENTS", false),
    autoIndex: bool(env, "AUTO_INDEX", true),
    autoIndexLimit: num(env, "AUTO_INDEX_LIMIT", 200, 1, 100000),
  };
}
