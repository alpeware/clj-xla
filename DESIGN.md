# DESIGN.md: Architectural Specification for `clj-xla`

## 1. Executive Overview & Philosophy

**`clj-xla`** is a high-performance, open-source Machine Learning compiler framework and runtime for Clojure targeting **Java 25**. It provides a pure functional DSL for defining neural network ops, compiling them to **StableHLO MLIR**, and executing them directly on accelerator hardware (CUDA GPUs, AMD ROCm, Google TPUs, and AVX/NEON CPUs) via OpenXLA's **PJRT C API** using modern **Java 25 Project Panama Foreign Function & Memory (FFM) API** (`java.lang.foreign`).

### Core Guiding Principles

1. **Data over Code (Homoiconic AST):** The intermediate representation (IR) is a pure, flat Single Static Assignment (SSA) EDN graph inspired by JAX’s `jaxpr`. It can be inspected, serialized, validated, and mutated as standard Clojure data.
2. **AI-Agent Native:** Because the AST is pure EDN governed by strict Malli schemas, AI coding agents can generate, validate, optimize, and synthesize model graphs deterministically without writing code strings or dealing with macro expansion pitfalls.
3. **Sub-Millisecond REPL Feedback via Caching:** In-process, zero-copy native execution combined with SHA-256 IR compilation caching allows developers to modify model logic at the REPL and re-execute on persistent GPU memory buffers without restarting the JVM, reloading weights, or re-triggering heavy XLA codegen for unchanged subgraphs.
4. **Clean Division of Labor:** Clojure owns high-level control loops, symbolic automatic differentiation, dynamic graph transformations, and weight management. OpenXLA handles memory layout, kernel fusion, auto-vectorization, register placement, and hardware-level codegen.

---

## 2. System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   Clojure Application Layer                 │
│  - Control Loops (e.g., DiffusionGemma Canvas, Sampling)    │
│  - Model Weight Management (.safetensors Panama mmap)       │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                 Layer 3: Tracing & DSL                      │
│  - Shadowed Operators (+, *, -, /, tanh, pow, etc.)         │
│  - Scalar Auto-Lifting & Symbolic Execution Tracing         │
│  - Trace-Time Control Flow vs Runtime Ops (xla/cond, while) │
└──────────────────────────────┬──────────────────────────────┘
                               │ Emits Pure EDN SSA Graph
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                 Layer 2: EDN IR & Transformations           │
│  - Malli Graph Validation & Algebraic DCE/Constant Folding  │
│  - Reverse-Mode Autodiff Engine (VJP & Cotangent Accum)    │
│  - StableHLO MLIR Text Generator & Module Formatter         │
└──────────────────────────────┬──────────────────────────────┘
                               │ Emits StableHLO MLIR
                               ▼
┌─────────────────────────────────────────────────────────────┐
│          Layer 1.5: Compiler Caching & Executable Manager   │
│  - SHA-256 IR Hash Lookup & In-Memory PjRtExecutable Cache  │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                 Layer 1: Native Runtime (PJRT)               │
│  - Java 25 Project Panama (java.lang.foreign) FFM bindings  │
│  - SymbolLookup for GetPjrtApi() dynamic plugin loading     │
│  - Off-Heap Arenas, MemorySegment & Async PjRtBuffer safety │
│  - OpenXLA Dynamic Backend (libpjrt_cuda.so / cpu / rocm)   │
└──────────────────────────────┘
```

---

## 3. Detailed Layer Specifications

### Layer 1: Native Runtime & Memory (`clj-xla.pjrt`)

Layer 1 handles foreign function invocation and off-heap memory management without C++ JNI boilerplate using Java 25's finalized `java.lang.foreign` API.

#### PJRT C API Interface & Dynamic Binding
* **Plugin Entrypoint:** Loads vendor dynamic shared objects (`libpjrt_cuda.so`, `libpjrt_cpu.so`, `libpjrt_rocm.so`) at runtime using `SymbolLookup.libraryLookup`.
* **API Function Resolution:** Resolves the C entrypoint `const PJRT_Api* GetPjrtApi()`. Using Panama `StructLayout` definitions, `clj-xla.pjrt` maps the returned `PJRT_Api` struct table to invoke API function pointers (`PJRT_Client_Create`, `PJRT_Client_Compile`, `PJRT_Buffer_FromHostBuffer`, `PJRT_LoadedExecutable_Execute`).

#### Off-Heap Memory Management (`Arena` & `MemorySegment`)
* **Zero-Copy Weight Offloading:** `.safetensors` model parameters are mapped directly into off-heap `MemorySegment` memory via `FileChannel.map` using a `Shared` or `Confined` Java 25 `Arena`.
* **Async Host-to-Device (H2D) Buffer Safety:** `PjRtBuffer` instances are populated from host memory using `PJRT_Buffer_FromHostBuffer`. To prevent premature deallocation during async DMA transfers, `Arena` lifetimes are synchronized with PJRT event completion handles or explicit safe closeable scopes.

#### Key Components:
* `PjRtClient`: Manages hardware device initialization, platform attributes, memory spaces, and compilation contexts.
* `PjRtLoadedExecutable`: Wraps compiled XLA hardware binary executables.
* `PjRtBuffer`: Opaque handle to device-allocated VRAM/SRAM memory buffers.

---

### Layer 2: Intermediate Representation (`clj-xla.stablehlo`)

Layer 2 defines the pure data specification for the computation graph. The graph represents flat Single Static Assignment (SSA) equations, avoiding nested AST trees that complicate variable reuse, constant folding, and backpropagation.

#### Jaxpr-Inspired EDN SSA Format

In `clj-xla`, all inputs, intermediate tensors, constants, and outputs are represented as explicit SSA variables. Scalar constants are auto-lifted into explicit `:stablehlo/constant` equations:

```clojure
{:name "gelu_block"
 :invars  [[:x {:type [:tensor [1 128 768] :f32]}]]
 :outvars [:y]
 :eqns    [{:op :stablehlo/constant :value 0.5          :outvars [:c0]}
           {:op :stablehlo/constant :value 1.0          :outvars [:c1]}
           {:op :stablehlo/constant :value 0.7978845608 :outvars [:c2]}
           {:op :stablehlo/constant :value 0.044715     :outvars [:c3]}
           {:op :stablehlo/constant :value 3.0          :outvars [:c_three]}
           {:op :stablehlo/power    :invars [:x :c_three] :outvars [:t0]}
           {:op :stablehlo/multiply :invars [:t0 :c3]    :outvars [:t1]}
           {:op :stablehlo/add      :invars [:x :t1]     :outvars [:t2]}
           {:op :stablehlo/multiply :invars [:t2 :c2]    :outvars [:t3]}
           {:op :stablehlo/tanh     :invars [:t3]        :outvars [:t4]}
           {:op :stablehlo/add      :invars [:t4 :c1]    :outvars [:t5]}
           {:op :stablehlo/multiply :invars [:x :t5]     :outvars [:t6]}
           {:op :stablehlo/multiply :invars [:t6 :c0]    :outvars [:y]}]}
```

#### Malli Schema Validation (`clj-xla.stablehlo.schema`)

```clojure
(def ElementTypeSchema
  [:enum :f16 :f32 :f64 :bf16 :i8 :i16 :i32 :i64 :pred])

(def TensorTypeSchema
  [:tuple [:= :tensor] [:vector :int] ElementTypeSchema])

(def VarBindingSchema
  [:tuple keyword? TensorTypeSchema])

(def EquationSchema
  [:map
   [:op keyword?]
   [:invars [:vector keyword?]]
   [:outvars [:vector keyword?]]
   [:value {:optional true} [or number? boolean? vector?]]
   [:attrs {:optional true} map?]])

(def GraphSchema
  [:map
   [:name string?]
   [:invars [:vector VarBindingSchema]]
   [:outvars [:vector keyword?]]
   [:eqns [:vector EquationSchema]]])
```

#### StableHLO MLIR Text Printing
`clj-xla.stablehlo` formats EDN SSA graphs into compliant MLIR textual module strings containing `func.func @main(...)` with dialect annotations (`stablehlo.dot_general`, `stablehlo.broadcast_in_dim`, `stablehlo.constant`, etc.), passed directly to `PJRT_Client_Compile`.

---

### Layer 3: Tracing & Mathematical DSL (`clj-xla.trace` & `clj-xla.tensor`)

Layer 3 allows developers and AI agents to write standard functional math expressions using shadowed Clojure operators.

#### Operator Shadowing & Auto-Lifting (`clj-xla.tensor`)
* Core operators (`+`, `*`, `-`, `/`, `pow`, `tanh`, `sqrt`, `dot-general`, `slice`, `reshape`, `transpose`) are defined in `clj-xla.tensor`.
* When called with raw numerical scalars or Clojure collections, values are automatically lifted into SSA constant equations.
* When executed inside a `trace` macro context, operations append equations to a thread-local SSA graph builder.

#### Trace-Time Meta-Control Flow vs Runtime Dynamic Control Flow
* **Trace-Time Control Flow:** Standard Clojure `if`, `when`, `cond`, `dotimes`, `loop`/`recur` execute during tracing to conditionally emit graph equations or unroll repetitive network layers (e.g. 24 Transformer blocks).
* **Runtime Dynamic Control Flow:** Dynamic runtime conditions or dynamic loops on GPU/TPU tensors are expressed using explicit higher-order operators (`xla/cond`, `xla/while_loop`, `xla/map`, `xla/reduce`).

#### Pure Clojure Kernel Example

```clojure
(ns clj-xla.example.kernels
  (:refer-clojure :exclude [+ * - / min max pow tanh sqrt])
  (:require [clj-xla.tensor :refer [+ * - / min max pow tanh sqrt reduce-mean]]))

(defn gelu [x]
  (let [c-sqrt 0.7978845608]
    (* 0.5 x (+ 1.0 (tanh (* c-sqrt (+ x (* 0.044715 (pow x 3.0)))))))))

(defn layer-norm [x gamma beta eps]
  (let [mean (reduce-mean x :axes [-1] :keep-dims true)
        diff (- x mean)
        var  (reduce-mean (pow diff 2.0) :axes [-1] :keep-dims true)
        std  (sqrt (+ var eps))
        norm (/ diff std)]
    (+ (* norm gamma) beta)))
```

---

### Layer 4: Higher-Order Transformations (`clj-xla.autodiff` & `clj-xla.opt`)

Layer 4 transforms forward graphs into backward graphs and optimizes graphs prior to MLIR serialization.

1. **Reverse-Mode Autodiff (VJPs):**
   * Traverses the forward SSA `:eqns` vector in reverse order.
   * Emits vector-Jacobian product (VJP) equations for backpropagation.
   * **Cotangent Accumulation:** When an intermediate tensor `v` is consumed by multiple downstream operations, `clj-xla.autodiff` automatically inserts a `:stablehlo/add` node to sum accumulated gradients (`dv = dv1 + dv2`) before propagating gradients backward.
   * **Shape Reduction:** Handles broadcast alignment during reverse propagation via automatic `reduce_sum` along broadcasted axes.
   * Merges forward and backward equations, appending optimizer updates (e.g., AdamW state updates).

2. **Frontend Graph Optimizations (`clj-xla.opt`):**
   * **Dead Code Elimination (DCE):** Prunes unused SSA nodes not transitively reachable from `:outvars`.
   * **Constant Folding:** Pre-computes purely scalar static subgraphs during tracing.
   * **`vmap` Vectorization:** Automatically maps batch dimensions over unbatched single-sample functions.

---

### Layer 1.5 & Execution Engine: SHA-256 Compilation Caching (`clj-xla.compile`)

To guarantee sub-millisecond REPL feedback while working with heavy XLA compiler backends:

1. **Graph Hashing:** When `trace-and-compile` is invoked, `clj-xla.compile` computes a SHA-256 hash of the normalized EDN graph (including input shapes and target hardware platform).
2. **In-Memory Executable Cache:** The compiled `PjRtLoadedExecutable` native handle is stored in an in-memory `atom` map.
3. **Execution Latency:**
   * **First Call (Cold):** Tracing (< 1ms) + StableHLO Printing (< 1ms) + XLA Codegen (50ms - 200ms) = ~50-200ms total.
   * **Subsequent Calls (Warm / REPL Re-eval):** SHA-256 Cache Hit -> Direct execution of `PjRtLoadedExecutable` on `PjRtBuffer` handles in **< 1ms**.

---

## 4. Target Use Case Architectures

### A. Non-Autoregressive Generation (DiffusionGemma)

* **Strategy:** Compile the heavy backbone denoising model into a static StableHLO executable.
* **Control Loop:** Pure Clojure code orchestrates the 256-token canvas, entropy estimation, and self-conditioning loops, invoking the compiled `PjRtLoadedExecutable` per denoising step on persistent `PjRtBuffer` handles in GPU memory.

### B. Fine-Tuning (FunctionGemma 270M / LoRA)

* **Strategy:** Trace forward pass $\to$ generate VJP backward pass via `clj-xla.autodiff` $\to$ append AdamW updates.
* **Execution:** Compile the full training step into a single XLA executable. XLA automatically manages activation checkpointing (rematerialization) to fit within GPU VRAM constraints.

### C. Distillation & Pre-Training

* **Strategy:** Execute two distinct PJRT executables concurrently:
  1. Teacher Model (Forward Pass Only, FP16/FP8).
  2. Student Model (Forward + Backward Pass).
* Loss function combines cross-entropy and KL divergence over logits directly in device memory.

---

## 5. Developer & AI Agent Workflows

### Human REPL Workflow

1. Load model weights into off-heap GPU buffers via `pjrt/to-device`.
2. Edit kernel logic or loss functions directly in your Clojure editor.
3. Evaluate the trace macro in the REPL (`(trace-and-compile my-kernel args)`).
4. Execute instantly (< 1ms warm execution latency) over existing device memory buffers without losing GPU state.

### AI Agent Protocol (RSI & Code Generation)

1. **Generation:** AI agent emits an EDN graph map (`{:invars [...], :eqns [...]}`).
2. **Validation:** Agent verifies the graph against `malli/validate` locally in microseconds.
3. **Execution & Feedback:** Agent passes the validated EDN graph to `clj-xla.core/compile-and-run`.
4. **Mutations:** Agent applies pure data transformations (`assoc-in`, `update`, `postwalk`) to explore novel network topologies or kernel optimizations deterministically.

---

## 6. Repository Layout & Phased Roadmap

```
clj-xla/
├── README.md
├── DESIGN.md
├── deps.edn
└── src/
    └── clj_xla/
        ├── core.clj           ;; High-level JIT execution API & REPL entrypoint
        ├── compile.clj        ;; Graph hashing & executable caching
        ├── pjrt.clj           ;; Panama Java 25 FFM bindings to libpjrt_cuda/cpu
        ├── stablehlo.clj      ;; EDN SSA schema, validation, & MLIR printer
        ├── tensor.clj         ;; Shadowed operators & scalar auto-lifting
        ├── trace.clj          ;; Symbolic tracing engine
        ├── autodiff.clj       ;; Reverse-mode VJP auto-differentiation & cotangent sum
        ├── opt.clj            ;; DCE & constant folding graph passes
        └── safetensors.clj    ;; Panama MemorySegment zero-copy off-heap weight loader
```

### Development Phases

* **Phase 1: Foundation (PJRT & Memory with Java 25 Panama)**
  * Implement Panama FFM bindings for `GetPjrtApi`, `PjRtClient`, `PjRtBuffer`, `PjRtLoadedExecutable`.
  * Build mmap `.safetensors` parser into off-heap `MemorySegment`s using Java 25 `Arena`.

* **Phase 2: StableHLO IR, Tracing Engine & Caching**
  * Implement strict Malli schemas for Jaxpr-style EDN graphs.
  * Build `clj-xla.tensor` shadowed operator library and symbolic tracer.
  * Write EDN-to-StableHLO MLIR text printer.
  * Implement SHA-256 graph hash compilation cache.

* **Phase 3: Autodiff & Optimizations**
  * Build reverse-mode VJP generator with cotangent accumulation and broadcast reduction.
  * Add DCE and constant-folding passes in `clj-xla.opt`.

* **Phase 4: High-Level Models & Agent Tooling**
  * Implement Gemma / FunctionGemma 270M fine-tuning pipelines.
  * Build DiffusionGemma discrete text diffusion canvas runtime.
  * Package agent schema validation tooling for autonomous graph synthesis.
