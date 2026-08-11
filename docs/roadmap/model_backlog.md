# clj-xla Model & Feature Backlog Matrix

This backlog maintains a clear differentiation between **currently supported models/features** and **upcoming planned models/features**.

---

## 🟢 Currently Supported (Production-Ready in `clj-xla`)

| Model Family / Feature | Target Architecture | Specification & Graph | Clojure Implementation Source | Verification Status |
| :--- | :--- | :--- | :--- | :--- |
| **GPT-2 (Small/Med/Large/XL)** | Decoder Causal LM (MHA) | [Spec & Graph](file:///home/simonpure/src/alpeware/clj-xla/docs/models/gpt2.md) ([GitHub](https://github.com/alpeware/clj-xla/blob/main/docs/models/gpt2.md)) | [`clj-xla.models.gpt2`](file:///home/simonpure/src/alpeware/clj-xla/src/clj_xla/models/gpt2.clj) ([GitHub](https://github.com/alpeware/clj-xla/blob/main/src/clj_xla/models/gpt2.clj)) | **Verified** (CPU, SYCL, ROCm) |
| **SmolLM (135M/360M/1.7B)** | Lightweight Edge LM (GQA/SwiGLU) | [Spec & Graph](file:///home/simonpure/src/alpeware/clj-xla/docs/models/smollm.md) ([GitHub](https://github.com/alpeware/clj-xla/blob/main/docs/models/smollm.md)) | [`clj-xla.models.smollm`](file:///home/simonpure/src/alpeware/clj-xla/src/clj_xla/models/smollm.clj) ([GitHub](https://github.com/alpeware/clj-xla/blob/main/src/clj_xla/models/smollm.clj)) | **Verified** (CPU, SYCL) |
| **Gemma 2 (2B/9B/27B)** | GQA, Gemma RMSNorm ($1+w$) | [Spec & Graph](file:///home/simonpure/src/alpeware/clj-xla/docs/models/gemma2.md) ([GitHub](https://github.com/alpeware/clj-xla/blob/main/docs/models/gemma2.md)) | [`clj-xla.models.gemma`](file:///home/simonpure/src/alpeware/clj-xla/src/clj_xla/models/gemma.clj) ([GitHub](https://github.com/alpeware/clj-xla/blob/main/src/clj_xla/models/gemma.clj)) | **Verified** (CPU, SYCL) |
| **Gemma 3 (1B/4B/12B/27B)** | Gemma 3 Architecture | [Spec & Graph](file:///home/simonpure/src/alpeware/clj-xla/docs/models/gemma3.md) ([GitHub](https://github.com/alpeware/clj-xla/blob/main/docs/models/gemma3.md)) | [`clj-xla.models.gemma`](file:///home/simonpure/src/alpeware/clj-xla/src/clj_xla/models/gemma.clj) ([GitHub](https://github.com/alpeware/clj-xla/blob/main/src/clj_xla/models/gemma.clj)) | **Verified** (CPU, SYCL) |
| **Gemma 4 (E2B/E4B)** | Gemma 4 Per-Layer Gating | [Spec & Graph](file:///home/simonpure/src/alpeware/clj-xla/docs/models/gemma4_e2b_e4b.md) ([GitHub](https://github.com/alpeware/clj-xla/blob/main/docs/models/gemma4_e2b_e4b.md)) | [`clj-xla.models.gemma`](file:///home/simonpure/src/alpeware/clj-xla/src/clj_xla/models/gemma.clj) ([GitHub](https://github.com/alpeware/clj-xla/blob/main/src/clj_xla/models/gemma.clj)) | **Verified** (CPU, Intel Arc SYCL) |
| **Safetensors INT8 Quantization** | Header JSON + Tensor Loading | [Wiki](file:///home/simonpure/src/alpeware/clj-xla/docs/wiki/quantization.md) ([GitHub](https://github.com/alpeware/clj-xla/blob/main/docs/wiki/quantization.md)) | [`clj-xla.safetensors`](file:///home/simonpure/src/alpeware/clj-xla/src/clj_xla/safetensors.clj) ([GitHub](https://github.com/alpeware/clj-xla/blob/main/src/clj_xla/safetensors.clj)) | **Verified** |
| **SOTA Multi-Backend Benchmarks** | Profiling Engine & CLI | [Overview](file:///home/simonpure/src/alpeware/clj-xla/docs/README.md) ([GitHub](https://github.com/alpeware/clj-xla/blob/main/docs/README.md)) | [`scripts.benchmark`](file:///home/simonpure/src/alpeware/clj-xla/scripts/benchmark.clj) ([GitHub](https://github.com/alpeware/clj-xla/blob/main/scripts/benchmark.clj)) | **Verified** (CPU, SYCL, ROCm) |

---

## 🟡 Planned Backlog (Upcoming Models & Features)

| Target Model / Feature | Specification & Plan | Priority | Target Namespace | Key Additions |
| :--- | :--- | :--- | :--- | :--- |
| **Gemma 4 26B-A4B MoE** | [Spec & Graph](file:///home/simonpure/src/alpeware/clj-xla/docs/models/gemma4_26b_a4b.md) ([GitHub](https://github.com/alpeware/clj-xla/blob/main/docs/models/gemma4_26b_a4b.md)) | **High** | `clj-xla.models.gemma4-moe` | 8-expert sparse MoE router (2 active experts, 3.9B active params) |
| **Meta Muse-Glimmer-30B** | [Spec & Graph](file:///home/simonpure/src/alpeware/clj-xla/docs/models/muse_glimmer_30b.md) ([GitHub](https://github.com/alpeware/clj-xla/blob/main/docs/models/muse_glimmer_30b.md)) | **High** | `clj-xla.models.muse-glimmer` | ViT-G/14 Vision Encoder, DFlash 16-token block speculative predictor, ATEM tool markup |
| **DeepSeek-V3 / MLA** | [Wiki](file:///home/simonpure/src/alpeware/clj-xla/docs/wiki/attention.md) ([GitHub](https://github.com/alpeware/clj-xla/blob/main/docs/wiki/attention.md)) | **High** | `clj-xla.nn.attention` | Multi-Head Latent Attention (576-dim latent vector compression) |
| **In-Graph INT4 / FP4 Dequant** | [Wiki](file:///home/simonpure/src/alpeware/clj-xla/docs/wiki/quantization.md) ([GitHub](https://github.com/alpeware/clj-xla/blob/main/docs/wiki/quantization.md)) | **High** | `clj-xla.nn.quantization` | StableHLO in-graph de-quantization fused into GEMM kernels |
| **EAGLE-2 Speculative Decoding** | [Wiki](file:///home/simonpure/src/alpeware/clj-xla/docs/wiki/speculative_decoding.md) ([GitHub](https://github.com/alpeware/clj-xla/blob/main/docs/wiki/speculative_decoding.md)) | **Medium** | `clj-xla.generation` | Tree-structured candidate draft verification pass |
