#!/usr/bin/env python3
"""
JAX Workload Benchmark Suite for clj-xla Parity & Performance Gap Verification.

Measures JAX JIT compilation latency, execution warmup, mean/P99 latencies,
and FLOPs/TFLOPS for standard ML kernels and transformer layer blocks across CPU and SYCL backends.
"""

import sys
import time
import json
import argparse
import numpy as np

def setup_jax_backend(backend_name):
    from jax._src.lib import xla_client
    import jax
    import jax.numpy as jnp

    target_dev = None
    if backend_name == "sycl":
        try:
            xla_client.load_pjrt_plugin_dynamically("sycl", "bin/libpjrt_sycl.so")
            client = xla_client.make_c_api_client("sycl", {"allocator": "platform"})
            target_dev = client.devices()[0]
        except Exception as e:
            print(f"Warning initializing SYCL C API client: {e}", flush=True)

    if target_dev is None:
        target_dev = jax.devices("cpu")[0]

    return jax, jnp, target_dev

def calculate_stats(latencies_ms):
    sorted_ms = sorted(latencies_ms)
    n = len(sorted_ms)
    p50_idx = int(n * 0.50)
    p90_idx = int(n * 0.90)
    p99_idx = int(min(n - 1, int(n * 0.99)))
    mean_ms = sum(sorted_ms) / n
    variance = sum((x - mean_ms) ** 2 for x in sorted_ms) / n
    std_dev_ms = variance ** 0.5
    return {
        "min_ms": round(sorted_ms[0], 3),
        "max_ms": round(sorted_ms[-1], 3),
        "mean_ms": round(mean_ms, 3),
        "std_dev_ms": round(std_dev_ms, 3),
        "p50_ms": round(sorted_ms[p50_idx], 3),
        "p90_ms": round(sorted_ms[p90_idx], 3),
        "p99_ms": round(sorted_ms[p99_idx], 3)
    }

def benchmark_fn(name, jit_fn, args, warmup_iters=5, measure_iters=50, flops=None):
    # Measure Cold JIT Compile Time
    t0 = time.perf_counter()
    res = jit_fn(*args)
    res.block_until_ready()
    t1 = time.perf_counter()
    cold_jit_ms = (t1 - t0) * 1000.0

    # Warmup Passes
    for _ in range(warmup_iters):
        res = jit_fn(*args)
        res.block_until_ready()

    # Measured Iterations
    latencies = []
    for _ in range(measure_iters):
        st = time.perf_counter()
        res = jit_fn(*args)
        res.block_until_ready()
        et = time.perf_counter()
        latencies.append((et - st) * 1000.0)

    stats = calculate_stats(latencies)
    stats["cold_jit_ms"] = round(cold_jit_ms, 3)

    if flops:
        mean_sec = stats["mean_ms"] / 1000.0
        tflops = (flops / mean_sec) / 1e12
        stats["tflops"] = round(tflops, 3)
    else:
        stats["tflops"] = 0.0

    return stats

def dev_put(tensor, jax, dev):
    return jax.device_put(tensor, dev)

# --- Workload Definitions in JAX ---

def make_gemm_fp32(jnp, jax, dev):
    m, k, n = 1024, 1024, 1024
    a = dev_put(jnp.ones((m, k), dtype=jnp.float32), jax, dev)
    b = dev_put(jnp.ones((k, n), dtype=jnp.float32), jax, dev)
    flops = 2 * m * n * k
    @jax.jit
    def run(a_t, b_t):
        return jnp.matmul(a_t, b_t)
    return "gemm-fp32", run, (a, b), flops

def make_gemm_bf16(jnp, jax, dev):
    m, k, n = 1024, 1024, 1024
    a = dev_put(jnp.ones((m, k), dtype=jnp.bfloat16), jax, dev)
    b = dev_put(jnp.ones((k, n), dtype=jnp.bfloat16), jax, dev)
    flops = 2 * m * n * k
    @jax.jit
    def run(a_t, b_t):
        return jnp.matmul(a_t, b_t)
    return "gemm-bf16", run, (a, b), flops

def make_rms_norm(jnp, jax, dev):
    b, s, d = 1, 2048, 4096
    x = dev_put(jnp.ones((b, s, d), dtype=jnp.float32), jax, dev)
    gamma = dev_put(jnp.ones((d,), dtype=jnp.float32), jax, dev)
    @jax.jit
    def run(x_t, g_t):
        ms = jnp.mean(x_t ** 2, axis=-1, keepdims=True)
        return (x_t / jnp.sqrt(ms + 1e-6)) * g_t
    return "rms-norm", run, (x, gamma), None

def make_swiglu(jnp, jax, dev):
    b, s, d = 1, 2048, 4096
    inter = 4096
    x = dev_put(jnp.ones((b, s, d), dtype=jnp.float32), jax, dev)
    w_gate = dev_put(jnp.ones((d, inter), dtype=jnp.float32), jax, dev)
    w_up = dev_put(jnp.ones((d, inter), dtype=jnp.float32), jax, dev)
    w_down = dev_put(jnp.ones((inter, d), dtype=jnp.float32), jax, dev)
    flops = 2 * (2 * b * s * d * inter + b * s * inter * d)
    @jax.jit
    def run(x_t, wg, wu, wd):
        gate = jax.nn.silu(jnp.matmul(x_t, wg))
        up = jnp.matmul(x_t, wu)
        return jnp.matmul(gate * up, wd)
    return "swiglu", run, (x, w_gate, w_up, w_down), flops

def make_gqa_causal_attn(jnp, jax, dev):
    b, seq, h_q, h_kv, d_k = 1, 128, 8, 1, 256
    q = dev_put(jnp.ones((b, seq, h_q, d_k), dtype=jnp.float32), jax, dev)
    k = dev_put(jnp.ones((b, seq, h_kv, d_k), dtype=jnp.float32), jax, dev)
    v = dev_put(jnp.ones((b, seq, h_kv, d_k), dtype=jnp.float32), jax, dev)
    o_w = dev_put(jnp.ones((h_q * d_k, h_q * d_k), dtype=jnp.float32), jax, dev)
    @jax.jit
    def run(q_t, k_t, v_t, ow):
        k_exp = jnp.repeat(k_t, h_q // h_kv, axis=2)
        v_exp = jnp.repeat(v_t, h_q // h_kv, axis=2)
        q_trans = jnp.transpose(q_t, (0, 2, 1, 3))
        k_trans = jnp.transpose(k_exp, (0, 2, 3, 1))
        v_trans = jnp.transpose(v_exp, (0, 2, 1, 3))
        scores = jnp.matmul(q_trans, k_trans) / (d_k ** 0.5)
        probs = jax.nn.softmax(scores, axis=-1)
        context = jnp.matmul(probs, v_trans)
        context_flat = jnp.reshape(jnp.transpose(context, (0, 2, 1, 3)), (b, seq, -1))
        return jnp.matmul(context_flat, ow)
    return "gqa-causal-attn", run, (q, k, v, o_w), None

def make_gpt2_block(jnp, jax, dev):
    b, s, d = 1, 128, 768
    x = dev_put(jnp.ones((b, s, d), dtype=jnp.float32), jax, dev)
    c_attn_w = dev_put(jnp.ones((d, 3 * d), dtype=jnp.float32), jax, dev)
    c_proj_w = dev_put(jnp.ones((d, d), dtype=jnp.float32), jax, dev)
    mlp_fc_w = dev_put(jnp.ones((d, 4 * d), dtype=jnp.float32), jax, dev)
    mlp_proj_w = dev_put(jnp.ones((4 * d, d), dtype=jnp.float32), jax, dev)
    flops = 2 * (b * s * d * (3 * d) + b * s * d * d + b * s * d * (4 * d) + b * s * (4 * d) * d)
    @jax.jit
    def run(x_t, c_attn, c_proj, mlp_fc, mlp_proj):
        ms1 = jnp.mean(x_t ** 2, axis=-1, keepdims=True)
        ln1 = x_t / jnp.sqrt(ms1 + 1e-5)
        qkv = jnp.matmul(ln1, c_attn)
        attn_out = jnp.matmul(qkv[:, :, :d], c_proj)
        res1 = x_t + attn_out
        ms2 = jnp.mean(res1 ** 2, axis=-1, keepdims=True)
        ln2 = res1 / jnp.sqrt(ms2 + 1e-5)
        fc = jax.nn.gelu(jnp.matmul(ln2, mlp_fc))
        mlp_out = jnp.matmul(fc, mlp_proj)
        return res1 + mlp_out
    return "gpt2-block", run, (x, c_attn_w, c_proj_w, mlp_fc_w, mlp_proj_w), flops

def make_gemma4_block(jnp, jax, dev):
    b, s, d = 1, 128, 1536
    x = dev_put(jnp.ones((b, s, d), dtype=jnp.float32), jax, dev)
    q_w = dev_put(jnp.ones((d, 2048), dtype=jnp.float32), jax, dev)
    k_w = dev_put(jnp.ones((d, 256), dtype=jnp.float32), jax, dev)
    v_w = dev_put(jnp.ones((d, 256), dtype=jnp.float32), jax, dev)
    o_w = dev_put(jnp.ones((2048, d), dtype=jnp.float32), jax, dev)
    gate_w = dev_put(jnp.ones((d, 6144), dtype=jnp.float32), jax, dev)
    up_w = dev_put(jnp.ones((d, 6144), dtype=jnp.float32), jax, dev)
    down_w = dev_put(jnp.ones((6144, d), dtype=jnp.float32), jax, dev)
    flops = 2 * (b * s * d * 2048 + b * s * d * 256 + b * s * d * 256 + b * s * 2048 * d + 2 * b * s * d * 6144 + b * s * 6144 * d)
    @jax.jit
    def run(x_t, qw, kw, vw, ow, gw, uw, dw):
        ms1 = jnp.mean(x_t ** 2, axis=-1, keepdims=True)
        ln1 = x_t / jnp.sqrt(ms1 + 1e-6)
        q = jnp.matmul(ln1, qw)
        k = jnp.matmul(ln1, kw)
        v = jnp.matmul(ln1, vw)
        attn_out = jnp.matmul(q, ow)
        res1 = x_t + attn_out
        ms2 = jnp.mean(res1 ** 2, axis=-1, keepdims=True)
        ln2 = res1 / jnp.sqrt(ms2 + 1e-6)
        gate = jax.nn.silu(jnp.matmul(ln2, gw))
        up = jnp.matmul(ln2, uw)
        mlp_out = jnp.matmul(gate * up, dw)
        return res1 + mlp_out
    return "gemma4-block", run, (x, q_w, k_w, v_w, o_w, gate_w, up_w, down_w), flops

def run_benchmarks_for_backend(backend_name):
    print(f"\n==========================================================================", flush=True)
    print(f"               Running JAX Benchmarks for Backend: [{backend_name}]        ", flush=True)
    print(f"==========================================================================", flush=True)
    jax, jnp, dev = setup_jax_backend(backend_name)
    print(f"Active JAX Device: {dev}", flush=True)

    workloads = [
        make_gemm_fp32(jnp, jax, dev),
        make_gemm_bf16(jnp, jax, dev),
        make_rms_norm(jnp, jax, dev),
        make_swiglu(jnp, jax, dev),
        make_gqa_causal_attn(jnp, jax, dev),
        make_gpt2_block(jnp, jax, dev),
        make_gemma4_block(jnp, jax, dev)
    ]
    results = {}
    print(f"{'Kernel':<18} | {'Mean (ms)':<10} | {'P99 (ms)':<10} | {'TFLOPS':<8} | {'Cold JIT (ms)':<12}", flush=True)
    print("-" * 70, flush=True)
    for name, fn, args, flops in workloads:
        stats = benchmark_fn(name, fn, args, flops=flops)
        stats["backend"] = str(backend_name)
        results[name] = stats
        print(f"{name:<18} | {stats['mean_ms']:<10.3f} | {stats['p99_ms']:<10.3f} | {stats['tflops']:<8.3f} | {stats['cold_jit_ms']:<12.3f}", flush=True)

    return results

def main():
    parser = argparse.ArgumentParser(description="JAX Benchmark Suite")
    parser.add_argument("--backend", "-b", choices=["auto", "cpu", "sycl", "all"], default="auto",
                        help="Target backend platform")
    args = parser.parse_args()

    backends = []
    if args.backend == "all":
        backends = ["sycl", "cpu"]
    elif args.backend == "auto":
        backends = ["sycl", "cpu"]
    else:
        backends = [args.backend]

    all_results = {}
    for b in backends:
        try:
            res = run_benchmarks_for_backend(b)
            all_results[b] = res
        except Exception as e:
            print(f"Error benchmarking JAX backend {b}: {e}")

    import os
    os.makedirs("target/benchmark-reports", exist_ok=True)
    report_path = "target/benchmark-reports/jax_benchmark_results.json"
    with open(report_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nJAX benchmark report saved to: {report_path}")

if __name__ == "__main__":
    main()
