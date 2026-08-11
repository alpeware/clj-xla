# SmolLM Architecture Specification & Verification

- **Status**: **Fully Supported** (SmolLM-135M, SmolLM-360M, SmolLM-1.7B)
- **Clojure Source**: [`src/clj_xla/models/smollm.clj`](file:///home/simonpure/src/alpeware/clj-xla/src/clj_xla/models/smollm.clj)

---

## 1. Model Configurations

| Variant | Hidden Dim | Layers | Query Heads | KV Heads | Head Dim |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **SmolLM-135M** | 576 | 30 | 9 | 3 | 64 |
| **SmolLM-360M** | 960 | 32 | 15 | 5 | 64 |
| **SmolLM-1.7B** | 2048 | 24 | 32 | 32 | 64 |

---

## 2. Key Components

1. **RMSNorm**: Standard root-mean-square normalization.
2. **GQA / MHA Attention**: RoPE embeddings with causal KV cache support.
3. **SwiGLU MLP**: Gated Feed-Forward projection.
