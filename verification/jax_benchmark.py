#!/usr/bin/env python3
"""
JAX Workload Benchmark Suite for clj-xla Parity & Performance Gap Verification.

Measures JAX JIT compilation latency, execution warmup, mean/P99 latencies,
and FLOPs/TFLOPS for standard ML kernels and transformer layer blocks.
"""

import time
import json
import numpy as np
import jax
import jax.numpy as jnp

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

# --- Workload Definitions in JAX ---

def make_gemm_fp32():
    m, k, n = 1024, 1024, 1024
    a = jnp.ones((m, k), dtype=jnp.float32)
    b = jnp.ones((k, n), dtype=jnp.float32)
    flops = 2 * m * n * k
    @jax.jit
    def run(a_t, b_t):
        return jnp.matmul(a_t, b_t)
    return "gemm-fp32", run, (a, b), flops

def make_gemm_bf16():
    m, k, n = 1024, 1024, 1024
    a = jnp.ones((m, k), dtype=jnp.bfloat16)
    b = jnp.ones((k, n), dtype=jnp.bfloat16)
    flops = 2 * m * n * k
    @jax.jit
    def run(a_t, b_t):
        return jnp.matmul(a_t, b_t)
    return "gemm-bf16", run, (a, b), flops

def make_rms_norm():
    b, s, d = 1, 2048, 4096
    x = jnp.ones((b, s, d), dtype=jnp.float32)
    gamma = jnp.ones((d,), dtype=jnp.float32)
    @jax.jit
    def run(x_t, g_t):
        ms = jnp.mean(x_t ** 2, axis=-1, keepdims=True)
        return (x_t / jnp.sqrt(ms + 1e-6)) * g_t
    return "rms-norm", run, (x, gamma), None

def make_swiglu():
    b, s, d = 1, 2048, 4096
    inter = 4096
    x = jnp.ones((b, s, d), dtype=jnp.float32)
    w_gate = jnp.ones((d, inter), dtype=jnp.float32)
    w_up = jnp.ones((d, inter), dtype=jnp.float32)
    w_down = jnp.ones((inter, d), dtype=jnp.float32)
    flops = 2 * (2 * b * s * d * inter + b * s * inter * d)
    @jax.jit
    def run(x_t, wg, wu, wd):
        gate = jax.nn.silu(jnp.matmul(x_t, wg))
        up = jnp.matmul(x_t, wu)
        return jnp.matmul(gate * up, wd)
    return "swiglu", run, (x, w_gate, w_up, w_down), flops

def make_gqa_causal_attn():
    b, seq, h_q, h_kv, d_k = 1, 128, 8, 1, 256
    q = jnp.ones((b, seq, h_q, d_k), dtype=jnp.float32)
    k = jnp.ones((b, seq, h_kv, d_k), dtype=jnp.float32)
    v = jnp.ones((b, seq, h_kv, d_k), dtype=jnp.float32)
    o_w = jnp.ones((h_q * d_k, h_q * d_k), dtype=jnp.float32)
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

def make_gpt2_block():
    b, s, d = 1, 128, 768
    x = jnp.ones((b, s, d), dtype=jnp.float32)
    c_attn_w = jnp.ones((d, 3 * d), dtype=jnp.float32)
    c_proj_w = jnp.ones((d, d), dtype=jnp.float32)
    mlp_fc_w = jnp.ones((d, 4 * d), dtype=jnp.float32)
    mlp_proj_w = jnp.ones((4 * d, d), dtype=jnp.float32)
    flops = 2 * (b * s * d * (3 * d) + b * s * d * d + b * s * d * (4 * d) + b * s * (4 * d) * d)
    @jax.jit
    def run(x_t, c_attn, c_proj, mlp_fc, mlp_proj):
        # Pre-LN Attn
        ms1 = jnp.mean(x_t ** 2, axis=-1, keepdims=True)
        ln1 = x_t / jnp.sqrt(ms1 + 1e-5)
        qkv = jnp.matmul(ln1, c_attn)
        attn_out = jnp.matmul(qkv[:, :, :d], c_proj)
        res1 = x_t + attn_out
        # Pre-LN MLP
        ms2 = jnp.mean(res1 ** 2, axis=-1, keepdims=True)
        ln2 = res1 / jnp.sqrt(ms2 + 1e-5)
        fc = jax.nn.gelu(jnp.matmul(ln2, mlp_fc))
        mlp_out = jnp.matmul(fc, mlp_proj)
        return res1 + mlp_out
    return "gpt2-block", run, (x, c_attn_w, c_proj_w, mlp_fc_w, mlp_proj_w), flops

def make_gemma4_block():
    b, s, d = 1, 128, 1536
    x = jnp.ones((b, s, d), dtype=jnp.float32)
    q_w = jnp.ones((d, 2048), dtype=jnp.float32)
    k_w = jnp.ones((d, 256), dtype=jnp.float32)
    v_w = jnp.ones((d, 256), dtype=jnp.float32)
    o_w = jnp.ones((2048, d), dtype=jnp.float32)
    gate_w = jnp.ones((d, 6144), dtype=jnp.float32)
    up_w = jnp.ones((d, 6144), dtype=jnp.float32)
    down_w = jnp.ones((6144, d), dtype=jnp.float32)
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

def main():
    print(f"JAX Device: {jax.devices()[0]}")
    workloads = [
        make_gemm_fp32(),
        make_gemm_bf16(),
        make_rms_norm(),
        make_swiglu(),
        make_gqa_causal_attn(),
        make_gpt2_block(),
        make_gemma4_block()
    ]
    results = {}
    print(f"{'Kernel':<18} | {'Mean (ms)':<10} | {'P99 (ms)':<10} | {'TFLOPS':<8} | {'Cold JIT (ms)':<12}")
    print("-" * 70)
    for name, fn, args, flops in workloads:
        stats = benchmark_fn(name, fn, args, flops=flops)
        results[name] = stats
        print(f"{name:<18} | {stats['mean_ms']:<10.3f} | {stats['p99_ms']:<10.3f} | {stats['tflops']:<8.3f} | {stats['cold_jit_ms']:<12.3f}")

    import os
    os.makedirs("target/benchmark-reports", exist_ok=True)
    report_path = "target/benchmark-reports/jax_benchmark_results.json"
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nJAX benchmark report saved to: {report_path}")

if __name__ == "__main__":
    main()
