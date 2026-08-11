# GPT-2 Architecture Specification & Verification

- **Status**: **Fully Supported**
- **Clojure Source**: [`src/clj_xla/models/gpt2.clj`](file:///home/simonpure/src/alpeware/clj-xla/src/clj_xla/models/gpt2.clj)
- **Test Suite**: [`test/clj_xla/models/gpt2_test.clj`](file:///home/simonpure/src/alpeware/clj-xla/test/clj_xla/models/gpt2_test.clj)

---

## 1. Architectural Overview

GPT-2 is a decoder-only causal language model utilizing standard Multi-Head Attention (MHA) and Conv1D / Linear projections.

| Hyperparameter | GPT-2 Small | GPT-2 Medium | GPT-2 Large | GPT-2 XL |
| :--- | :--- | :--- | :--- | :--- |
| **Parameters** | 124M | 355M | 774M | 1.5B |
| **Hidden Dimension ($d_{model}$)** | 768 | 1024 | 1280 | 1600 |
| **Layers ($N_{layers}$)** | 12 | 24 | 36 | 48 |
| **Attention Heads ($H$)** | 12 | 16 | 20 | 25 |
| **Vocabulary Size ($V$)** | 50,257 | 50,257 | 50,257 | 50,257 |

---

## 2. Key Neural Network Components

1. **LayerNorm**: Pre-LayerNorm placement (`ln_1` before self-attention, `ln_2` before MLP).
2. **Causal Self-Attention**: Standard MHA with triangular causal masking.
3. **GELU MLP**: 2-layer projection ($d_{model} \to 4 d_{model} \to d_{model}$) with GELU activation.

---

## 3. Verification & Execution

* **Cold JIT Compile Verification**:
  ```bash
  clojure -M:test -e "(require '[clj-xla.models.gpt2-test]) (clojure.test/run-tests 'clj-xla.models.gpt2-test)"
  ```
* **Benchmark Profile**: Included in [`scripts/benchmark.clj`](file:///home/simonpure/src/alpeware/clj-xla/scripts/benchmark.clj) as `gpt2-block`.
