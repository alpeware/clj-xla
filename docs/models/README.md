# Model Architecture Specifications & Mermaid Trace Graphs

This directory contains individual architectural specification documents and visual Mermaid graph representations for all supported and planned model families in `clj-xla`.

---

## 🟢 Supported Models (Production-Ready Codebase & Test Verification)

1. **[GPT-2 (Small / Medium / Large / XL)](file:///home/simonpure/src/alpeware/clj-xla/docs/models/gpt2.md)** ([GitHub](https://github.com/alpeware/clj-xla/blob/main/docs/models/gpt2.md))
   - Classic causal autoregressive decoder with Multi-Head Attention (MHA) and Conv1D projections.
   - Code: [`clj-xla.models.gpt2`](file:///home/simonpure/src/alpeware/clj-xla/src/clj_xla/models/gpt2.clj) ([GitHub](https://github.com/alpeware/clj-xla/blob/main/src/clj_xla/models/gpt2.clj)).

2. **[SmolLM (135M / 360M / 1.7B)](file:///home/simonpure/src/alpeware/clj-xla/docs/models/smollm.md)** ([GitHub](https://github.com/alpeware/clj-xla/blob/main/docs/models/smollm.md))
   - Lightweight edge models featuring Grouped-Query Attention (GQA) and SwiGLU activations.
   - Code: [`clj-xla.models.smollm`](file:///home/simonpure/src/alpeware/clj-xla/src/clj_xla/models/smollm.clj) ([GitHub](https://github.com/alpeware/clj-xla/blob/main/src/clj_xla/models/smollm.clj)).

3. **[Gemma 2 (2B / 9B / 27B)](file:///home/simonpure/src/alpeware/clj-xla/docs/models/gemma2.md)** ([GitHub](https://github.com/alpeware/clj-xla/blob/main/docs/models/gemma2.md))
   - Gemma RMSNorm ($+1.0$ weight offset), GQA, and SwiGLU MLP blocks.
   - Code: [`clj-xla.models.gemma`](file:///home/simonpure/src/alpeware/clj-xla/src/clj_xla/models/gemma.clj) ([GitHub](https://github.com/alpeware/clj-xla/blob/main/src/clj_xla/models/gemma.clj)).

4. **[Gemma 3 (1B / 4B / 12B / 27B)](file:///home/simonpure/src/alpeware/clj-xla/docs/models/gemma3.md)** ([GitHub](https://github.com/alpeware/clj-xla/blob/main/docs/models/gemma3.md))
   - Compact multi-modal ready architectures with 1B lightweight edge configuration.
   - Code: [`clj-xla.models.gemma`](file:///home/simonpure/src/alpeware/clj-xla/src/clj_xla/models/gemma.clj) ([GitHub](https://github.com/alpeware/clj-xla/blob/main/src/clj_xla/models/gemma.clj)).

5. **[Gemma 4 (E2B / E4B)](file:///home/simonpure/src/alpeware/clj-xla/docs/models/gemma4_e2b_e4b.md)** ([GitHub](https://github.com/alpeware/clj-xla/blob/main/docs/models/gemma4_e2b_e4b.md))
   - Gemma 4 architecture with per-layer scalar gating projections, Gemma RMSNorm, and GQA.
   - Code: [`clj-xla.models.gemma`](file:///home/simonpure/src/alpeware/clj-xla/src/clj_xla/models/gemma.clj) ([GitHub](https://github.com/alpeware/clj-xla/blob/main/src/clj_xla/models/gemma.clj)) & CLI [`scripts/gemma4_inference.clj`](file:///home/simonpure/src/alpeware/clj-xla/scripts/gemma4_inference.clj) ([GitHub](https://github.com/alpeware/clj-xla/blob/main/scripts/gemma4_inference.clj)).

---

## 🟡 Planned Backlog Models (Specs & Roadmap)

6. **[Gemma 4 26B-A4B Sparse MoE](file:///home/simonpure/src/alpeware/clj-xla/docs/models/gemma4_26b_a4b.md)** ([GitHub](https://github.com/alpeware/clj-xla/blob/main/docs/models/gemma4_26b_a4b.md))
   - Sparse Mixture-of-Experts (MoE) architecture with 26B total parameters and 4B active parameters per token.

7. **[Meta Muse-Glimmer-30B](file:///home/simonpure/src/alpeware/clj-xla/docs/models/muse_glimmer_30b.md)** ([GitHub](https://github.com/alpeware/clj-xla/blob/main/docs/models/muse_glimmer_30b.md))
   - Multimodal agentic model featuring DFlash 16-token parallel block speculative decoding, ~1.8B ViT-G/14 vision encoder, and ATEM tool markup.
