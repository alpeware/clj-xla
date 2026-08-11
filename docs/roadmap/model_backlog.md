# clj-xla Model & Feature Backlog Matrix

This backlog maintains a clear differentiation between **currently supported models/features** and **upcoming planned models/features**.

---

## 🟢 Currently Supported (Production-Ready in `clj-xla`)

| Model Family / Feature | Target Architecture | Implementation Path | Verification Status |
| :--- | :--- | :--- | :--- |
| **GPT-2 (Small/Med/Large/XL)** | Decoder Causal LM (MHA) | [`clj-xla.models.gpt2`](file:///home/simonpure/src/alpeware/clj-xla/src/clj_xla/models/gpt2.clj) | **Verified** (CPU, SYCL, ROCm) |
| **Gemma 2 / 3 / 4 (E2B/E4B)** | GQA, Gemma RMSNorm, Gated Residuals | [`clj-xla.models.gemma`](file:///home/simonpure/src/alpeware/clj-xla/src/clj_xla/models/gemma.clj) | **Verified** (CPU, Intel Arc SYCL) |
| **SmolLM (135M/360M/1.7B)** | Lightweight Edge LM (GQA/SwiGLU) | [`clj-xla.models.smollm`](file:///home/simonpure/src/alpeware/clj-xla/src/clj_xla/models/smollm.clj) | **Verified** (CPU, SYCL) |
| **Safetensors INT8 Quantization** | Header JSON + Tensor Deserialization | [`clj-xla.safetensors`](file:///home/simonpure/src/alpeware/clj-xla/src/clj_xla/safetensors.clj) | **Verified** |
| **SOTA Multi-Backend Benchmark Suite** | Performance & Throughput Profiling | [`scripts.benchmark`](file:///home/simonpure/src/alpeware/clj-xla/scripts/benchmark.clj) | **Verified** (CPU, SYCL, ROCm) |
| **Subprocess Isolation Harness** | C++ Teardown Crash Protection | [`clj-xla.test.isolated-runner`](file:///home/simonpure/src/alpeware/clj-xla/src/clj_xla/test/isolated_runner.clj) | **Verified** |

---

## 🟡 Planned Backlog (Upcoming Features & Models)

| Target Model / Feature | Priority | Target Namespace | Milestones / Key Additions |
| :--- | :--- | :--- | :--- |
| **Meta Muse-Glimmer-30B** | **High** | `clj-xla.models.muse-glimmer` | ViT-G/14 Vision Encoder, DFlash 16-token parallel block predictor, ATEM tool parser |
| **DeepSeek-V3 / MLA** | **High** | `clj-xla.nn.attention` | Multi-Head Latent Attention (576-dim latent KV cache vector compression) |
| **In-Graph INT4 / FP4 Dequantization** | **High** | `clj-xla.nn.quantization` | StableHLO in-graph de-quantization fused into tensor core GEMM kernels |
| **EAGLE-2 Speculative Decoding** | **Medium** | `clj-xla.generation` | Tree-structured candidate draft verification pass |
| **Qwen 2.5 / Qwen 3.6 MoE** | **Medium** | `clj-xla.models.qwen` | Sparse Mixture-of-Experts routing & top-k expert dispatch |
