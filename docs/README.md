# clj-xla Architectural Documentation & LLM Knowledge Base

Welcome to the **`clj-xla` Architectural Documentation & LLM Knowledge Base**.

`clj-xla` is a high-performance, pure Clojure numerical computing and deep learning framework built on **OpenXLA** and **StableHLO MLIR** via the Java Panama Foreign Function & Memory (FFM) API. It provides pure functional tensor abstractions, graph tracing, automatic differentiation, and hardware acceleration across CPU, Intel SYCL, AMD ROCm, and NVIDIA CUDA devices—with **Zero Java Escape Hatches** (Pure XLA compilation).

---

## 📚 Table of Contents

1. **[LLM Algorithm Wiki](file:///home/simonpure/src/alpeware/clj-xla/docs/wiki/README.md)**
   - Theoretical concepts, mathematical formulas, and reference Clojure implementations for attention mechanisms, normalizations, activations, speculative decoding, and quantization.
2. **[Model Architecture Specifications](file:///home/simonpure/src/alpeware/clj-xla/docs/models/)**
   - Architectural deep dives and verification specifications for supported and planned model families.
   - [GPT-2 Architecture Spec](file:///home/simonpure/src/alpeware/clj-xla/docs/models/gpt2.md) *(Supported)*
   - [Gemma 2 / 3 / 4 Architecture Spec](file:///home/simonpure/src/alpeware/clj-xla/docs/models/gemma.md) *(Supported)*
   - [SmolLM Architecture Spec](file:///home/simonpure/src/alpeware/clj-xla/docs/models/smollm.md) *(Supported)*
   - [Meta Muse-Glimmer-30B Architecture Spec & Plan](file:///home/simonpure/src/alpeware/clj-xla/docs/models/muse_glimmer.md) *(Planned)*
3. **[Model & Feature Backlog](file:///home/simonpure/src/alpeware/clj-xla/docs/roadmap/model_backlog.md)**
   - Matrix of currently supported models, features, hardware backends, and upcoming backlog items.
4. **[Full-Stack Consumer Hardware Roadmap](file:///home/simonpure/src/alpeware/clj-xla/docs/roadmap/full_stack_roadmap.md)**
   - Long-term vision spanning Inference, Distillation, Evals, Fine-Tuning (SFT/LoRA), and Pre-training on consumer hardware using OpenXLA.

---

## 📊 Current Status at a Glance

| Feature / Model | Status | Pure Clojure Source | Hardware Verification |
| :--- | :--- | :--- | :--- |
| **GPT-2 (Small/Med/Large/XL)** | **Supported** | [`clj-xla.models.gpt2`](file:///home/simonpure/src/alpeware/clj-xla/src/clj_xla/models/gpt2.clj) | CPU / SYCL / ROCm |
| **Gemma 2 / 3 / 4 (E2B, E4B)** | **Supported** | [`clj-xla.models.gemma`](file:///home/simonpure/src/alpeware/clj-xla/src/clj_xla/models/gemma.clj) | CPU / SYCL (Intel Arc) |
| **SmolLM (135M / 360M / 1.7B)** | **Supported** | [`clj-xla.models.smollm`](file:///home/simonpure/src/alpeware/clj-xla/src/clj_xla/models/smollm.clj) | CPU / SYCL |
| **INT8 Quantized Weights** | **Supported** | [`clj-xla.safetensors`](file:///home/simonpure/src/alpeware/clj-xla/src/clj_xla/safetensors.clj) | CPU / SYCL / ROCm |
| **Multi-Backend Benchmarks** | **Supported** | [`scripts.benchmark`](file:///home/simonpure/src/alpeware/clj-xla/scripts/benchmark.clj) | CPU / SYCL / ROCm / CUDA |
| **Meta Muse-Glimmer-30B** | **Planned** | `clj-xla.models.muse-glimmer` | Backlog |
| **DeepSeek-V3 / MLA** | **Planned** | `clj-xla.nn.attention` (MLA extension) | Backlog |
| **DFlash 16-Token Speculation** | **Planned** | `clj-xla.benchmark.runner` (DFlash) | Backlog |
| **In-Graph INT4/FP4 Dequant** | **Planned** | `clj-xla.nn.quantization` | Backlog |
