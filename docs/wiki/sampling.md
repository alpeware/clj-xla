# Autoregressive Sampling Algorithms & KV Cache Management

During autoregressive generation, output logits $z \in \mathbb{R}^V$ are converted into probability distributions and sampled to select the next token $x_{t+1}$.

---

## 1. Sampling Algorithms

### A. Temperature Scaling
Modulates distribution entropy by dividing raw logits by temperature $T > 0$:
$$z'_i = \frac{z_i}{T}$$
* $T < 1.0$: Sharpens distribution (more deterministic).
* $T > 1.0$: Flattens distribution (more creative).

### B. Top-K Filtering
Restricts sampling candidates to the top $K$ highest-probability logits, setting all other logits to $-\infty$:
$$z'_i = \begin{cases} z_i & \text{if } \text{rank}(z_i) \le K \\ -\infty & \text{otherwise} \end{cases}$$

### C. Top-P (Nucleus) Sampling
Accumulates candidate token probabilities until the cumulative sum reaches threshold $P \in (0, 1]$:
$$\sum_{i \in S} \text{softmax}(z)_i \ge P$$

---

## 2. KV-Cache Management

The Key-Value (KV) cache stores historical key and value tensors across steps to avoid re-computing attention for past tokens:
```
Step t-1:  KV_cache = [K_{0..t-1}, V_{0..t-1}]
Step t:    Compute [K_t, V_t] for new token x_t
           Update: KV_cache = concat(KV_cache, [K_t, V_t])
```

---

## 3. Clojure Implementation in `clj-xla`

- **Temperature Scaling**: [`clj-xla.sampling/apply-temperature`](file:///home/simonpure/src/alpeware/clj-xla/src/clj_xla/sampling.clj#L10) ([GitHub](https://github.com/alpeware/clj-xla/blob/main/src/clj_xla/sampling.clj#L10)).
- **Top-K Filtering**: [`clj-xla.sampling/apply-top-k`](file:///home/simonpure/src/alpeware/clj-xla/src/clj_xla/sampling.clj#L25) ([GitHub](https://github.com/alpeware/clj-xla/blob/main/src/clj_xla/sampling.clj#L25)).
- **Autoregressive Loop**: [`clj-xla.generation/autoregressive-generate`](file:///home/simonpure/src/alpeware/clj-xla/src/clj_xla/generation.clj#L15) ([GitHub](https://github.com/alpeware/clj-xla/blob/main/src/clj_xla/generation.clj#L15)).
- **KV Cache Invariants**: Tested in [`test/clj_xla/generation_test.clj`](file:///home/simonpure/src/alpeware/clj-xla/test/clj_xla/generation_test.clj) ([GitHub](https://github.com/alpeware/clj-xla/blob/main/test/clj_xla/generation_test.clj)).
