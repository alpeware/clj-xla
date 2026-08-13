# Debugging, Profiling, and Tracing in `clj-xla`

This guide outlines the debugging, profiling, and telemetry tracing tools built into **`clj-xla`** (inspired by **JAX** debugging affordances). These tools enable autonomous AI agents and software engineers to introspect execution graphs, diagnose numerical instability, profile execution latencies, and export performance spans for automated optimization feedback.

---

## 1. 🔍 Debugging Affordances (JAX Baseline Parity)

`clj-xla.debug` provides functional assertions and location metadata annotations that mirror JAX's `jax.debug` and `checkify` modules.

### A. Location Metadata (`with-xla-metadata`)
In large multi-layer architectures like Gemma 4 ($35$ layers), locating which specific layer or matrix multiplication triggered an issue can be difficult in raw MLIR text. `with-xla-metadata` attaches scoped location labels (`loc("gemma/layer_12/attn_matmul")`) to every traced equation:

```clojure
(require '[clj-xla.debug :as debug]
         '[clj-xla.tensor :as t])

(defn gemma-layer
  [layer-id x weights]
  (debug/with-xla-metadata {:op-name (str "gemma/layer_" layer-id "/attn")}
    (let [qkv (t/dot-general x (:qkv-w weights) ...)]
      qkv)))
```

When serialized via `clj-xla.stablehlo/graph->mlir-text`, instructions inherit exact source labels:
```mlir
%t_dot_12 = "stablehlo.dot_general"(%x, %qkv_w) { ... } : (tensor<1x1x768xf32>, tensor<768x2304xf32>) -> tensor<1x1x2304xf32> loc("gemma/layer_12/attn")
```

### B. Functional Assertions (`check-non-nan` & `check-non-inf`)
To catch exploding gradients or activation NaNs without crashing native OpenXLA executables, `check-non-nan` instruments the MLIR graph with runtime validation assertions:

```clojure
(defn safe-attention-softmax
  [logits]
  (let [probs (t/softmax logits)
        _ (debug/check-non-nan probs "NaN detected in attention softmax probabilities")]
    probs))
```

---

## 2. ⏱️ High-Precision Telemetry Profiling (`clj-xla.profile`)

`clj-xla.profile` provides microsecond-resolution span profiling for tracing, compilation, memory allocation, prefill, and decode step latencies.

### A. Micro-second Profile Spans (`with-profile`)
```clojure
(require '[clj-xla.profile :as profile])

(let [metrics (atom {})]
  (profile/with-profile metrics "graph_compilation"
    (compile-inference-executables session prompt-len))
  (println @metrics))
```

**Output Telemetry Report (`scratch/gemma4_profile.edn`)**:
```clojure
{:weight_transfer {:duration-ms 2809.12, :duration-us 2809127}
 :graph_compilation {:duration-ms 3573.13, :duration-us 3573137}
 :autoregressive_generation {:duration-ms 13167.43, :duration-us 13167437}}
```

### B. Chrome Trace & Perfetto Visual Export (`export-chrome-trace`)
Profile spans can be exported directly into standard Chrome Tracing JSON format (`chrome://tracing` or [`ui.perfetto.dev`](https://ui.perfetto.dev)):

```clojure
(let [spans-atom (atom [])]
  (binding [profile/*active-trace-spans* spans-atom]
    (run-agent-workload ...))
  (profile/save-chrome-trace! @spans-atom "scratch/gemma4_chrome_trace.json"))
```

---

## 3. 📈 Automated Performance Observations & Recommendations for Gemma 4

Based on empirical profile metrics collected from Gemma 4 E2B inference runs:

| Phase | Observed Latency | Bottleneck Analysis | SOTA Optimization Recommendation |
| :--- | :--- | :--- | :--- |
| **Weight Transfer** | **2,809 ms** (2.81 s) | Cold host-to-device PCIe allocation for 4.6 GB parameters | **Pre-allocated Permanent VRAM Session**: Keep weights pinned in device memory across agent turns. |
| **Graph Compilation** | **3,573 ms** (3.57 s) | Runtime JIT compilation of prefill + decode graphs | **Persistent OpenXLA Cache**: Enable `--xla_gpu_kernel_cache_file` to reuse compiled GCN binaries across restarts. |
| **Prefill Latency** | **835 ms** (17 tok) | Single-pass prompt processing | **Grouped GEMM & INT8 Weights**: Quantize static weights to INT8 to double memory bandwidth throughput. |
| **Decode Step Latency** | **422 ms / tok** (CPU) / **36 ms / tok** (GPU) | Host-side token step dispatch loop | **In-VRAM Fused `stablehlo.while` Loop**: Eliminate per-token host invocations by running the entire decode loop inside GPU VRAM. |
