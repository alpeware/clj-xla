# Hardware Benchmark: AMD Ryzen Desktop (AMD Radeon RX 7900 XTX 24GB ROCm)

- **System Model**: AMD Desktop Workstation
- **CPU**: AMD Ryzen 9 9950X High-Performance Desktop CPU (32 Threads, AVX-512)
- **GPU**: AMD Radeon RX 7900 XTX (24 GB VRAM - RDNA 3 Architecture)
- **PCIe Interface**: PCIe 4.0 x16
- **OS / Linux Kernel**: Gentoo Linux (Kernel 6.x, x86_64)
- **Driver Telemetry**:
  - AMD ROCm Driver Version: `6.0.0` / `7.2.0`
  - OpenXLA PJRT Plugin: `bin/libpjrt_rocm.so` (API Version 24.0)

---

## 1. Empirical Results Matrix (`clj-xla`: Host CPU vs. AMD Radeon RX 7900 XTX ROCm)

*Execution parameters: `--warmup 5 --measure 50` via `./scripts/benchmark.sh`.*

| Workload Kernel | `clj-xla` CPU Mean (ms) | `clj-xla` ROCm GPU Mean (ms) | ROCm GPU P99 (ms) | ROCm GPU TFLOPS / Bandwidth | GPU Speedup Factor |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **GEMM FP32 ($1024^3$)** | 1.30 ms | **0.46 ms** | 0.60 ms | 4.63 TFLOPS | **2.80x** |
| **GEMM BF16 ($1024^3$)** | 1.74 ms | **0.12 ms** | 0.15 ms | 17.47 TFLOPS | **14.15x** |
| **RMSNorm ($1 \times 2048 \times 4096$)** | 7.50 ms | **0.18 ms** | 0.27 ms | 365.88 GB/s | **40.99x** |
| **SwiGLU Activation ($1 \times 2048 \times 4096$)** | 262.47 ms | **97.67 ms** | 101.51 ms | 2.81 TFLOPS | **2.69x** |
| **GQA Causal Attention ($1 \times 128 \times 8 \times 256$)** | 8.53 ms | **0.72 ms** | 0.88 ms | 1.48 TFLOPS | **11.78x** |
| **GPT-2 Layer Block ($1 \times 128 \times 768$)** | 5.61 ms | **0.49 ms** | 0.54 ms | 3.71 TFLOPS | **11.50x** |
| **Gemma 4 Layer Block ($1 \times 128 \times 1536$)** | 16.20 ms | **2.99 ms** | 3.82 ms | 3.23 TFLOPS | **5.42x** |

---

## 2. Python/XLA (Official AMD `rocm/jax:latest` Container) vs. JVM/XLA (`clj-xla`) Matrix

*Comparison collected on the AMD Radeon RX 7900 XTX 24GB GPU running official AMD `rocm/jax:latest` container vs native `clj-xla` OpenXLA PJRT C API backend.*

| Workload Kernel | Official AMD JAX `rocm/jax` Mean | `clj-xla` ROCm GPU Mean | `clj-xla` ROCm GPU P50 | Official JAX TFLOPS | `clj-xla` ROCm TFLOPS |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **GEMM FP32 ($1024^3$)** | **0.146 ms** | 0.464 ms | 0.468 ms | **14.71 TFLOPS** | 4.63 TFLOPS |
| **GEMM BF16 ($1024^3$)** | **0.078 ms** | 0.123 ms | 0.122 ms | **27.53 TFLOPS** | 17.47 TFLOPS |
| **RMSNorm ($1 \times 2048 \times 4096$)** | **0.092 ms** | 0.183 ms | 0.191 ms | **$727.8\text{ GB/s}$** | $365.88\text{ GB/s}$ |
| **SwiGLU Activation ($1 \times 2048 \times 4096$)** | **19.138 ms** | 97.669 ms | 97.396 ms | **28.73 TFLOPS** | 2.81 TFLOPS |
| **GQA Causal Attention ($1 \times 128 \times 8 \times 256$)** | **0.332 ms** | 0.724 ms | 0.704 ms | **7.28 TFLOPS** | 1.48 TFLOPS |
| **GPT-2 Layer Block ($1 \times 128 \times 768$)** | **0.288 ms** | 0.488 ms | 0.485 ms | **6.29 TFLOPS** | 3.71 TFLOPS |
| **Gemma 4 Layer Block ($1 \times 128 \times 1536$)** | **0.881 ms** | 2.989 ms | 2.805 ms | **10.28 TFLOPS** | 3.23 TFLOPS |

---

## 3. Reproduction Commands

- **Run `clj-xla` ROCm Hardware Benchmark**:
  ```bash
  ./scripts/benchmark.sh --backend rocm --warmup 5 --measure 50
  ```
- **Run Python JAX Benchmark in Official AMD ROCm Container**:
  ```bash
  docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video \
    --security-opt seccomp=unconfined -v $(pwd):/workspace -w /workspace \
    rocm/jax:latest python3 verification/jax_benchmark.py --backend rocm
  ```
