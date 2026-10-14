import { TransformersAnswerer } from "../src/models.ts";

const id = process.argv[2] ?? "onnx-community/Qwen2.5-0.5B-Instruct";
const a = new TransformersAnswerer(id);
const t = Date.now();
const out = await a.answer("How much VRAM does my laptop have?", [
  "My laptop has 24GB RAM and a GTX 1650 with 4GB VRAM.",
  "The work PC has CPU-only Qwen2.5-Coder-7B at about 4 tokens per second.",
]);
console.log(`${id} (${Date.now() - t} ms):\n${out}`);
