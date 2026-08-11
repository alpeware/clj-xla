# LLM Architecture & Algorithm Reference Wiki

This Wiki serves as an open, educational reference for state-of-the-art LLM architectures, mathematical formulations, and optimization algorithms. Each topic pairs theoretical principles with the pure Clojure / StableHLO MLIR reference implementation in `clj-xla`.

---

## 📖 Wiki Topics

### 1. [Attention Mechanisms](file:///home/simonpure/src/alpeware/clj-xla/docs/wiki/attention.md)
- **Multi-Head Attention (MHA)**: Standard query-key-value self-attention.
- **Grouped-Query Attention (GQA)**: Shared KV heads for reduced memory bandwidth overhead.
- **Rotary Position Embeddings (RoPE)**: Relative position encoding via complex plane rotations.
- **Gemma 4 Hybrid Sliding Window**: Local window attention eviction combined with global layers.
- **DeepSeek Multi-Head Latent Attention (MLA)**: Low-rank KV cache vector compression.
- **FlashDecoding & Split-K**: Parallel single-query ($B=1$) KV reduction on GPU.

### 2. [Normalization Variants](file:///home/simonpure/src/alpeware/clj-xla/docs/wiki/normalization.md)
- **Standard LayerNorm**: Mean and variance scaling with gain ($\gamma$) and bias ($\beta$).
- **Root Mean Square Normalization (RMSNorm)**: Mean-square variance scaling without mean subtraction.
- **Gemma RMSNorm**: $+1.0$ weight offset variant: $y = \frac{x}{\text{RMS}(x)} \times (1.0 + w)$.

### 3. [Activation Functions](file:///home/simonpure/src/alpeware/clj-xla/docs/wiki/activations.md)
- **GELU / SiLU / ReLU**: Standard non-linearities.
- **SwiGLU & GeGLU**: Gated Linear Unit feed-forward sub-blocks.

### 4. [Speculative Decoding Algorithms](file:///home/simonpure/src/alpeware/clj-xla/docs/wiki/speculative_decoding.md)
- **Draft-Model Speculative Decoding**: Autoregressive candidate proposal with target verification.
- **EAGLE / EAGLE-2 & Medusa**: Multi-head / feature-level speculation without separate draft models.
- **Meta DFlash Block Speculation**: 16-token parallel block prediction.
- **Prompt-Lookup / N-gram Speculation**: Contextual string matching for zero-overhead candidate draft proposals.

### 5. [Low-Bit Quantization](file:///home/simonpure/src/alpeware/clj-xla/docs/wiki/quantization.md)
- **Post-Training Quantization (PTQ)**: INT8, INT4, AWQ, and GPTQ.
- **In-Graph De-quantization**: Fusing integer-to-float weight conversion with GEMM matrix multiplication in StableHLO MLIR.
