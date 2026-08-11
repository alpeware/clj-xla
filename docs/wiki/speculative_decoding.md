# Speculative Decoding Algorithms

Speculative decoding mitigates the memory-bandwidth bottleneck of single-request ($B=1$) autoregressive inference by predicting multiple candidate tokens and verifying them in a single target model forward pass.

---

## 1. Algorithmic Overview

```
                      ┌───────────────────────────────────────┐
                      │  Small Draft Model / Predictor Head   │
                      └──────────────────┬────────────────────┘
                                         │ Proposes K candidate tokens
                                         ▼
                      ┌───────────────────────────────────────┐
                      │   Large Target Model Parallel Pass    │
                      └──────────────────┬────────────────────┘
                                         │ Verifies K tokens in 1 memory load
                                         ▼
                      Accepted Tokens: [T1, T2, T3] (+ 1 new token)
```

---

## 2. Speculative Decoding Variants

### A. Draft-Model Speculative Decoding
- **Approach**: Uses a small auxiliary model (e.g. 1B parameter draft model for a 70B target model) sharing the same tokenizer.
- **Verification**: The target model runs a batched forward pass over the $K$ candidate tokens.

### B. Medusa & EAGLE / EAGLE-2
- **Approach**: Replaces the separate draft model with lightweight extra prediction heads attached directly to the target model's top hidden state.
- **EAGLE-2**: Uses tree-structured candidate verification to achieve $2\times - 3\times$ speedups at $B=1$.

### C. Meta DFlash (Block-Level Speculative Decoding)
- **Approach**: Predicts a **block of 16 tokens in a single parallel forward pass** (instead of sequential token generation).
- **Target**: Used in Meta's **Muse-Glimmer-30B** for local agentic task completion.

### D. Prompt-Lookup / N-Gram Speculation (Lossless)
- **Approach**: Performs fast string matching on existing conversation/prompt history to guess candidate tokens.
- **Zero Overhead**: Requires zero extra parameters or auxiliary models.
