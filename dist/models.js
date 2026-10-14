/** Memoises a loader but forgets failures, so a transient download error can be retried. */
function lazy(load) {
    let pending;
    return () => {
        pending ??= load().catch((error) => {
            pending = undefined;
            throw error;
        });
        return pending;
    };
}
function unitRows(tensor) {
    const [count = 0, dim = 0] = tensor.dims;
    const out = [];
    for (let i = 0; i < count; i++) {
        const row = Float32Array.from({ length: dim }, (_, d) => tensor.data[i * dim + d]);
        out.push(row);
    }
    return out;
}
export class TransformersEmbedder {
    signature;
    #config;
    #load;
    constructor(config) {
        this.#config = config;
        this.signature = [config.embedModel, config.embedPooling, config.embedDtype, config.queryPrefix, config.documentPrefix].join("|");
        this.#load = lazy(async () => {
            const { pipeline } = await import("@huggingface/transformers");
            const extractor = await pipeline("feature-extraction", config.embedModel, { dtype: config.embedDtype });
            return extractor;
        });
    }
    async embedDocuments(texts) {
        if (!texts.length)
            return [];
        const extract = await this.#load();
        const prefixed = texts.map((t) => this.#config.documentPrefix + t);
        return unitRows(await extract(prefixed, { pooling: this.#config.embedPooling, normalize: true }));
    }
    async embedQuery(text) {
        const extract = await this.#load();
        const [row] = unitRows(await extract([this.#config.queryPrefix + text], { pooling: this.#config.embedPooling, normalize: true }));
        if (!row)
            throw new Error("Embedding model returned no vector.");
        return row;
    }
}
/** A cross-encoder: reads the query and a passage together, so it orders candidates better than vectors alone. */
export class TransformersReranker {
    id;
    #load;
    constructor(modelId) {
        this.id = modelId;
        this.#load = lazy(async () => {
            const { AutoTokenizer, AutoModelForSequenceClassification } = await import("@huggingface/transformers");
            const tokenizer = (await AutoTokenizer.from_pretrained(modelId));
            const model = (await AutoModelForSequenceClassification.from_pretrained(modelId, {
                dtype: "q8",
            }));
            return { tokenizer, model };
        });
    }
    async score(query, passages) {
        if (!passages.length)
            return [];
        const { tokenizer, model } = await this.#load();
        const inputs = tokenizer(new Array(passages.length).fill(query), {
            text_pair: [...passages],
            padding: true,
            truncation: true,
            max_length: 512,
        });
        const { logits } = await model(inputs);
        const width = logits.dims[1] ?? 1;
        return passages.map((_, i) => width === 1 ? logits.data[i] : logits.data[i * width + width - 1] - logits.data[i * width]);
    }
}
/** Optional: writes a short answer from the matching passages with a small local chat model. */
export class TransformersAnswerer {
    id;
    #load;
    constructor(modelId) {
        this.id = modelId;
        this.#load = lazy(async () => {
            const { pipeline } = await import("@huggingface/transformers");
            const generator = await pipeline("text-generation", modelId, { dtype: "q4" });
            return generator;
        });
    }
    async answer(query, passages) {
        const generate = await this.#load();
        const context = passages.map((p, i) => `[${i + 1}] ${p}`).join("\n\n");
        const messages = [
            {
                role: "system",
                content: "Answer the question using only the numbered excerpts from past conversations. " +
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
