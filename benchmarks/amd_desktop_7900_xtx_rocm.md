# Hardware Benchmark: AMD Ryzen Desktop (AMD Radeon RX 7900 XTX 24GB ROCm)

- **System Model**: AMD Desktop Workstation
- **CPU**: AMD Ryzen High-Performance Desktop CPU
- **GPU**: AMD Radeon RX 7900 XTX (24 GB VRAM - RDNA 3 Architecture)
- **PCIe Interface**: PCIe 4.0 x16
- **OS / Linux Kernel**: Linux 6.x
- **Driver Telemetry**:
  - AMD ROCm Driver Version: `7.2.0` / `6.0.0`
  - OpenXLA PJRT Plugin: `bin/libpjrt_rocm.so` (API Version 24.0)

---

## 1. Empirical Results Matrix (`clj-xla`: Host CPU vs. AMD Radeon RX 7900 XTX ROCm)

*Note: Run `clojure -M scripts/benchmark.clj --backend rocm` on the AMD Desktop to populate empirical execution figures.*

| Workload Kernel | `clj-xla` CPU Mean (ms) | `clj-xla` ROCm GPU Mean (ms) | ROCm GPU P99 (ms) | ROCm GPU TFLOPS / Bandwidth | GPU Speedup Factor |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **GEMM FP32 ($1024^3$)** | TBD | **TBD** | TBD | TBD TFLOPS | **TBD** |
| **GEMM BF16 ($1024^3$)** | TBD | **TBD** | TBD | TBD TFLOPS | **TBD** |
| **RMSNorm ($1 \times 2048 \times 4096$)** | TBD | **TBD** | TBD | Memory Bandwidth | **TBD** |
| **SwiGLU Activation ($1 \times 2048 \times 4096$)** | TBD | **TBD** | TBD | TBD TFLOPS | **TBD** |
| **GQA Causal Attention ($1 \times 128 \times 8 \times 256$)** | TBD | **TBD** | TBD | TBD TFLOPS | **TBD** |
| **GPT-2 Layer Block ($1 \times 128 \times 768$)** | TBD | **TBD** | TBD | TBD TFLOPS | **TBD** |
| **Gemma 4 Layer Block ($1 \times 128 \times 1536$)** | TBD | **TBD** | TBD | TBD TFLOPS | **TBD** |

---

## 2. Reproduction Commands

- **Run `clj-xla` ROCm Hardware Benchmark**:
  ```bash
  clojure -M scripts/benchmark.clj --backend rocm --warmup 5 --measure 50
  ```
- **Run Python JAX Verification Script**:
  ```bash
  verification/.venv/bin/python verification/jax_benchmark.py
  ```
