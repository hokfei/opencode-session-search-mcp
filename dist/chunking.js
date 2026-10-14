/** Renders a session as the text that gets chunked, indexed and searched. */
export function buildTranscript(meta, messages) {
    const parts = [`Title: ${meta.title || "untitled"}`];
    for (const m of messages) {
        const text = m.text.trim();
        if (text)
            parts.push(`${m.role}: ${text}`);
    }
    return parts.join("\n\n");
}
/**
 * Splits text into overlapping windows. A window ends at a paragraph, line or word
 * boundary when one exists in its last fifth, so chunks rarely cut a word in half.
 */
export function chunkText(text, size, overlap) {
    const clean = text.trim();
    if (!clean)
        return [];
    if (size <= 0)
        throw new RangeError("size must be positive");
    const keep = Math.min(Math.max(0, overlap), Math.floor(size / 2));
    const chunks = [];
    let start = 0;
    while (start < clean.length) {
        let end = Math.min(start + size, clean.length);
        if (end < clean.length) {
            const floor = start + Math.floor(size * 0.8);
            const window = clean.slice(floor, end);
            const cut = Math.max(window.lastIndexOf("\n\n"), window.lastIndexOf("\n"), window.lastIndexOf(" "));
            if (cut >= 0)
                end = floor + cut + 1;
        }
        const piece = clean.slice(start, end).trim();
        if (piece)
            chunks.push(piece);
        if (end >= clean.length)
            break;
        start = Math.max(end - keep, start + 1);
    }
    return chunks;
}
