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
| **GEMM FP32 ($1024^3$)** | 1.19 ms | **0.45 ms** | 0.67 ms | 4.80 TFLOPS | **2.64x** |
| **GEMM BF16 ($1024^3$)** | 1.58 ms | **0.15 ms** | 0.18 ms | 14.47 TFLOPS | **10.53x** |
| **RMSNorm ($1 \times 2048 \times 4096$)** | 4.04 ms | **0.21 ms** | 0.47 ms | 323.22 GB/s | **19.24x** |
| **SwiGLU Activation ($1 \times 2048 \times 4096$)** | 240.49 ms | **103.34 ms** | 105.44 ms | 2.66 TFLOPS | **2.33x** |
| **GQA Causal Attention ($1 \times 128 \times 8 \times 256$)** | 4.56 ms | **0.77 ms** | 2.95 ms | 1.39 TFLOPS | **5.92x** |
| **GPT-2 Layer Block ($1 \times 128 \times 768$)** | 3.61 ms | **0.51 ms** | 0.59 ms | 3.58 TFLOPS | **7.08x** |
| **Gemma 4 Layer Block ($1 \times 128 \times 1536$)** | 15.49 ms | **2.66 ms** | 2.73 ms | 3.63 TFLOPS | **5.82x** |

---

## 2. Python/XLA (Official AMD `rocm/jax:latest` Container) vs. JVM/XLA (`clj-xla`) Matrix

*Comparison collected on the AMD Radeon RX 7900 XTX 24GB GPU running official AMD `rocm/jax:latest` container vs native `clj-xla` OpenXLA PJRT C API backend.*

| Workload Kernel | Official AMD JAX `rocm/jax` Mean | `clj-xla` ROCm GPU Mean | `clj-xla` ROCm GPU P50 | Official JAX TFLOPS | `clj-xla` ROCm TFLOPS |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **GEMM FP32 ($1024^3$)** | **0.146 ms** | 0.447 ms | 0.437 ms | **14.71 TFLOPS** | 4.80 TFLOPS |
| **GEMM BF16 ($1024^3$)** | **0.078 ms** | 0.148 ms | 0.145 ms | **27.53 TFLOPS** | 14.47 TFLOPS |
| **RMSNorm ($1 \times 2048 \times 4096$)** | **0.092 ms** | 0.208 ms | 0.185 ms | **$727.8\text{ GB/s}$** | $323.22\text{ GB/s}$ |
| **SwiGLU Activation ($1 \times 2048 \times 4096$)** | **19.138 ms** | 103.342 ms | 103.371 ms | **28.73 TFLOPS** | 2.66 TFLOPS |
| **GQA Causal Attention ($1 \times 128 \times 8 \times 256$)** | **0.332 ms** | 0.774 ms | 0.725 ms | **7.28 TFLOPS** | 1.39 TFLOPS |
| **GPT-2 Layer Block ($1 \times 128 \times 768$)** | **0.288 ms** | 0.506 ms | 0.503 ms | **6.29 TFLOPS** | 3.58 TFLOPS |
| **Gemma 4 Layer Block ($1 \times 128 \times 1536$)** | **0.881 ms** | **2.662 ms** | 2.667 ms | **10.28 TFLOPS** | **3.63 TFLOPS** |

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
