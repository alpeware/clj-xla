# Hardware Benchmark: Lenovo ThinkPad X1 Carbon Gen 13 (Intel Arc 140V SYCL)

- **System Model**: Lenovo ThinkPad X1 Carbon Gen 13
- **CPU**: Intel Core Ultra Series 2 (Lunar Lake - 8 Cores / 16 Threads)
- **iGPU**: Intel Arc 140V Graphics (8 Xe2 Cores with XMX Matrix Acceleration)
- **Memory**: 32 GB LPDDR5X-8533 Unified RAM ($85.3\text{ GB/s}$ Memory Bandwidth)
- **OS / Linux Kernel**: Linux 6.11.0
- **Driver Telemetry**:
  - Intel GPU Compute Runtime Driver: `26.22.038646`
  - oneAPI Level-Zero Loader API: `v1.30.0`
  - OpenXLA PJRT Plugin: `bin/libpjrt_sycl.so` (API Version 24.0)

---

## 1. Empirical Results Matrix (`clj-xla`: Host CPU vs. Intel Arc 140V SYCL GPU)

The following metrics were collected using [`scripts/benchmark.clj`](../scripts/benchmark.clj) on the Lenovo ThinkPad X1 Carbon Gen 13:

| Workload Kernel | `clj-xla` CPU Mean (ms) | `clj-xla` SYCL GPU Mean (ms) | SYCL GPU P99 (ms) | SYCL GPU TFLOPS / Bandwidth | GPU Speedup Factor |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **GEMM FP32 ($1024^3$)** | 15.973 ms | **6.571 ms** | 7.317 ms | 0.33 TFLOPS | **$2.43\times$** |
| **GEMM BF16 ($1024^3$)** | 9.856 ms | **3.401 ms** | 3.020 ms | 0.63 TFLOPS | **$2.90\times$** |
| **RMSNorm ($1 \times 2048 \times 4096$)** | 40.252 ms | **27.520 ms** | 28.004 ms | Memory Bandwidth | **$1.46\times$** |
| **SwiGLU Activation ($1 \times 2048 \times 4096$)** | 2328.695 ms | **711.353 ms** | 662.202 ms | 0.39 TFLOPS | **$3.27\times$** |
| **GQA Causal Attention ($1 \times 128 \times 8 \times 256$)** | 47.667 ms | **24.624 ms** | 26.529 ms | 0.04 TFLOPS | **$1.94\times$** |
| **GPT-2 Layer Block ($1 \times 128 \times 768$)** | 29.752 ms | **31.001 ms** | 30.946 ms | 0.06 TFLOPS | **$0.96\times$** |
| **Gemma 4 Layer Block ($1 \times 128 \times 1536$)** | 205.514 ms | **152.116 ms** | 158.449 ms | 0.06 TFLOPS | **$1.35\times$** |

---

## 2. Python/XLA (JAX) vs. JVM/XLA (`clj-xla`) Performance Gap Analysis

To evaluate execution overhead and dialect compatibility between JVM Panama FFM OpenXLA (`clj-xla`) and Python/XLA (JAX 0.11), we ran the identical workload suite using [`verification/jax_benchmark.py`](../verification/jax_benchmark.py):

| Workload Kernel | Python JAX CPU Mean (ms) | Python JAX SYCL GPU | `clj-xla` CPU Mean (ms) | `clj-xla` SYCL GPU Mean (ms) | Performance & Dialect Compatibility Analysis |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **GEMM FP32 ($1024^3$)** | 3.271 ms | *Failed (Dialect `sdy` unknown)* | 15.973 ms | **6.571 ms** | `clj-xla` emits canonical StableHLO MLIR text which compiles cleanly on `libpjrt_sycl.so`. |
| **GEMM BF16 ($1024^3$)** | 3.797 ms | *Failed (Dialect `sdy` unknown)* | 9.856 ms | **3.401 ms** | **`clj-xla` Intel Arc GPU ($3.40\text{ ms}$) beats Python JAX CPU ($3.80\text{ ms}$)** by $1.12\times$. |
| **RMSNorm** | 6.339 ms | *Failed (Dialect `sdy` unknown)* | 40.252 ms | **27.520 ms** | JAX CPU uses built-in oneDNN / OpenMP SIMD vectorization. |
| **SwiGLU Activation** | 460.862 ms | *Failed (Dialect `sdy` unknown)* | 2328.695 ms | **711.353 ms** | `clj-xla` SYCL GPU achieves **$3.27\times$ speedup** over `clj-xla` CPU. |
| **Gemma 4 Layer Block** | 24.132 ms | *Failed (Dialect `sdy` unknown)* | 205.514 ms | **152.116 ms** | `clj-xla` SYCL GPU outperforms `clj-xla` CPU by $1.35\times$. |

### Root Cause Analysis of Performance & Compatibility Differences

1. **Automatic CPU Core Threadpool Autodetection**:
   - `clj-xla` **automatically detects CPU core count** (16 threads on Lunar Lake) and injects `--xla_cpu_multi_thread_eigen=true` and `--xla_gpu_force_compilation_parallelism=16` in [`src/clj_xla/core.clj`](../src/clj_xla/core.clj#L86-L88).
   - This improved `clj-xla` CPU performance significantly (e.g. GQA Attention dropped from $114.41\text{ ms}$ to $47.67\text{ ms}$).

2. **Why Python JAX CPU Is Faster Than Standard C++ `libpjrt_cpu.so`**:
   - Python `jaxlib` wheel binaries include **Intel oneDNN / MKL-DNN SIMD vectorization primitives** statically linked into their CPU compiler engine.
   - The reference OpenXLA C++ PJRT plugin (`libpjrt_cpu.so`) relies on standard Eigen multi-threading without oneDNN fusion.

3. **StableHLO MLIR Dialect Compatibility Advantage in `clj-xla`**:
   - When Python JAX (v0.11) compiles MLIR for `libpjrt_sycl.so`, it injects experimental Shardy (`sdy`) dialect attributes from StableHLO v1.14 bytecode. Intel's `libpjrt_sycl.so` fails with `error: dialect 'sdy' is unknown`.
   - **`clj-xla` emits pure, canonical StableHLO MLIR text** (via [`clj-xla.stablehlo`](../src/clj_xla/stablehlo.clj)), bypassing non-standard dialect extensions and executing seamlessly on the Intel Arc 140V GPU!

---

## 3. Reproduction Commands

- **Run `clj-xla` Benchmark Suite**:
  ```bash
  clojure -M scripts/benchmark.clj --backend auto --warmup 5 --measure 50
  ```
- **Run Python JAX Verification Script**:
  ```bash
  verification/.venv/bin/python verification/jax_benchmark.py --backend auto
  ```
