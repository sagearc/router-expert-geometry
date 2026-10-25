# Fixed public-model titles

`public-titles256.json` preserves the 256 Wikipedia-derived title strings from
the retained OLMoE 5k capture, including their order and spelling.
The source artifact SHA-256 is
`ac9b99a619600b4032ea0a5cb7650547ce08bbc060008c989ddb2502797e0e3f`.
Its metadata identifies `wiki_routing_subcats_step39750.npz` as the input.

This is an alphabetically ordered title set, including category-like names,
not random Wikipedia prose or a representative language-evaluation sample.

```bash
uv run --extra public-models prepare-public-titles --model olmoe --out titles.npz
# Use --model qwen for the Qwen tokenizer.
```

Preparation downloads pinned tokenizer files, saves token IDs, lengths and
provenance, and rejects empty sequences. OLMoE produces 1,339 non-padding tokens;
Qwen produces 1,224. Model checkpoint selection is separate from tokenization.
Token counts alone do not establish historical token-ID identity; see
[public-rank validation](../validation/public-rank.md) and
[OLMoE validation](../validation/public-olmoe.md).
