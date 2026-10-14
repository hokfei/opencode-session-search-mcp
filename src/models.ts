import type { Config } from "./config.ts";
import type { Answerer, Embedder, Reranker } from "./types.ts";

interface TensorLike {
  readonly data: ArrayLike<number>;
  readonly dims: readonly number[];
}

type Extractor = (texts: string[], options: { pooling: string; normalize: boolean }) => Promise<TensorLike>;

/** Memoises a loader but forgets failures, so a transient download error can be retried. */
function lazy<T>(load: () => Promise<T>): () => Promise<T> {
  let pending: Promise<T> | undefined;
  return () => {
    pending ??= load().catch((error: unknown) => {
      pending = undefined;
      throw error;
    });
    return pending;
  };
}

function unitRows(tensor: TensorLike): Float32Array[] {
  const [count = 0, dim = 0] = tensor.dims;
  const out: Float32Array[] = [];
  for (let i = 0; i < count; i++) {
    const row = Float32Array.from({ length: dim }, (_, d) => tensor.data[i * dim + d]!);
    out.push(row);
  }
  return out;
}

export class TransformersEmbedder implements Embedder {
  readonly signature: string;
  readonly #config: Config;
  readonly #load: () => Promise<Extractor>;

  constructor(config: Config) {
    this.#config = config;
    this.signature = [config.embedModel, config.embedPooling, config.embedDtype, config.queryPrefix, config.documentPrefix].join("|");
    this.#load = lazy(async () => {
      const { pipeline } = await import("@huggingface/transformers");
      const extractor = await pipeline("feature-extraction", config.embedModel, { dtype: config.embedDtype as "q8" });
      return extractor as unknown as Extractor;
    });
  }

  async embedDocuments(texts: readonly string[]): Promise<Float32Array[]> {
    if (!texts.length) return [];
    const extract = await this.#load();
    const prefixed = texts.map((t) => this.#config.documentPrefix + t);
    return unitRows(await extract(prefixed, { pooling: this.#config.embedPooling, normalize: true }));
  }

  async embedQuery(text: string): Promise<Float32Array> {
    const extract = await this.#load();
    const [row] = unitRows(await extract([this.#config.queryPrefix + text], { pooling: this.#config.embedPooling, normalize: true }));
    if (!row) throw new Error("Embedding model returned no vector.");
    return row;
  }
}

type Tokenizer = (a: string[], options: Record<string, unknown>) => unknown;
type SequenceModel = (inputs: unknown) => Promise<{ logits: TensorLike }>;

/** A cross-encoder: reads the query and a passage together, so it orders candidates better than vectors alone. */
export class TransformersReranker implements Reranker {
  readonly id: string;
  readonly #load: () => Promise<{ tokenizer: Tokenizer; model: SequenceModel }>;

  constructor(modelId: string) {
    this.id = modelId;
    this.#load = lazy(async () => {
      const { AutoTokenizer, AutoModelForSequenceClassification } = await import("@huggingface/transformers");
      const tokenizer = (await AutoTokenizer.from_pretrained(modelId)) as unknown as Tokenizer;
      const model = (await AutoModelForSequenceClassification.from_pretrained(modelId, {
        dtype: "q8",
      })) as unknown as SequenceModel;
      return { tokenizer, model };
    });
  }

  async score(query: string, passages: readonly string[]): Promise<number[]> {
    if (!passages.length) return [];
    const { tokenizer, model } = await this.#load();
    const inputs = tokenizer(new Array<string>(passages.length).fill(query), {
      text_pair: [...passages],
      padding: true,
      truncation: true,
      max_length: 512,
    });
    const { logits } = await model(inputs);
    const width = logits.dims[1] ?? 1;
    return passages.map((_, i) =>
      width === 1 ? logits.data[i]! : logits.data[i * width + width - 1]! - logits.data[i * width]!,
    );
  }
}

interface GenerationOutput {
  readonly generated_text: ReadonlyArray<{ role: string; content: string }> | string;
}

type Generator = (messages: unknown, options: Record<string, unknown>) => Promise<GenerationOutput[]>;

/** Optional: writes a short answer from the matching passages with a small local chat model. */
export class TransformersAnswerer implements Answerer {
  readonly id: string;
  readonly #load: () => Promise<Generator>;

  constructor(modelId: string) {
    this.id = modelId;
    this.#load = lazy(async () => {
      const { pipeline } = await import("@huggingface/transformers");
      const generator = await pipeline("text-generation", modelId, { dtype: "q4" as "q8" });
      return generator as unknown as Generator;
    });
  }

  async answer(query: string, passages: readonly string[]): Promise<string> {
    const generate = await this.#load();
    const context = passages.map((p, i) => `[${i + 1}] ${p}`).join("\n\n");
    const messages = [
      {
        role: "system",
        content:
          "Answer the question using only the numbered excerpts from past conversations. " +
          "If they do not contain the answer, say so. Be brief and cite excerpt numbers.",
      },
      { role: "user", content: `Excerpts:\n${context}\n\nQuestion: ${query}` },
    ];
    const [out] = await generate(messages, { max_new_tokens: 300, do_sample: false });
    const text = out?.generated_text;
    const reply = Array.isArray(text) ? text.at(-1)?.content : text;
    return (reply ?? "").trim();
  }
}
