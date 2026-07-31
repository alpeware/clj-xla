# DESIGN.md: Architectural Specification for `clj-xla`

## 1. Executive Overview & Philosophy

**`clj-xla`** is a high-performance, open-source Machine Learning compiler framework and runtime for Clojure. It provides a pure functional DSL for defining neural network ops, compiling them to **StableHLO MLIR**, and executing them directly on hardware (CUDA GPUs, AMD ROCm, Google TPUs, and AVX/NEON CPUs) via OpenXLA's **PJRT C API** using modern **Java 21+ Project Panama** (`java.lang.foreign`).

### Core Guiding Principles

1. **Data over Code (Homoiconic AST):** The intermediate representation (IR) is a pure, flat Single Static Assignment (SSA) EDN graph inspired by JAX’s `jaxpr`. It can be inspected, serialized, validated, and mutated as standard Clojure data.
2. **AI-Agent Native:** Because the AST is pure EDN governed by strict schemas (Malli), AI coding agents can generate, validate, and optimize model graphs deterministically without handling complex code-string generation or macro pitfalls.
3. **Sub-50ms REPL Feedback Loop:** In-process, zero-copy native execution allows developers to alter graph logic at the REPL and re-execute on persistent GPU memory buffers without restarting the JVM or reloading model weights from disk.
4. **Clean Division of Labor:** Clojure owns high-level control loops, symbolic automatic differentiation, and graph transformations. OpenXLA handles memory layout, kernel fusion, register placement, and hardware-level codegen.

---

## 2. System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   Clojure Application Layer                 │
│  - Control Loops (e.g., DiffusionGemma Canvas, Sampling)    │
│  - Model Weight Management (.safetensors mmap)              │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                 Layer 3: Tracing & DSL                      │
│  - Shadowed Operators (+, *, -, /, tanh, pow, etc.)         │
│  - Scalar Auto-Lifting & Symbolic Execution Tracing         │
└──────────────────────────────┬──────────────────────────────┘
                               │ Emits Pure EDN SSA Graph
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                 Layer 2: EDN IR & Transformations           │
│  - Malli Graph Validation & Algebraic DCE/Constant Folding  │
│  - Reverse-Mode Autodiff Engine (VJP Generation)           │
│  - StableHLO MLIR Text / Bytecode Serializer                │
└──────────────────────────────┬──────────────────────────────┘
                               │ Emits StableHLO MLIR
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                 Layer 1: Native Runtime (PJRT)               │
│  - Java 21+ Project Panama (java.lang.foreign) bindings     │
│  - Zero-Copy Off-Heap Arenas & PjRtBuffer handles           │
│  - OpenXLA Dynamic Backend (libpjrt_cuda.so / cpu / rocm)   │
└─────────────────────────────────────────────────────────────┘

```

---

## 3. Detailed Layer Specifications

### Layer 1: Native Runtime & Memory (`clj-xla.pjrt`)

Layer 1 handles foreign function invocation and off-heap memory management without C++ JNI boilerplate.

* **PJRT C API Interface:** Interoperates with `pjrt_c_api.h` via `java.lang.foreign.Linker` and `FunctionDescriptor`. Loads vendor-specific plugins (`libpjrt_cuda.so`, `libpjrt_cpu.so`, `libpjrt_rocm.so`) dynamically at runtime.
* **Memory Management via Project Panama (`Arena` & `MemorySegment`):**
* `.safetensors` weight files are mmap'd directly into off-heap `MemorySegment` buffers using a `Shared` or `Confined` `Arena`.
* Weights are passed to `PjRtBuffer` instances via zero-copy host pointers without touching the JVM garbage-collected heap.


* **Key Components:**
* `PjRtClient`: Manages device initialization, memory spaces, and compilation contexts.
* `PjRtLoadedExecutable`: Wraps compiled XLA hardware executables.
* `PjRtBuffer`: Opaque handle to device-allocated VRAM buffers.



---

### Layer 2: Intermediate Representation (`clj-xla.stablehlo`)

Layer 2 defines the pure data specification for the computation graph. The graph represents flat Single Static Assignment (SSA) equations, avoiding nested AST trees that complicate variable reuse and backprop.

#### Jaxpr-Inspired EDN SSA Format

```clojure
{:name "gelu_block"
 :invars  [[:x {:type [:tensor [1 128 768] :f32]}]]
 :outvars [:y]
 :consts  {:c0 0.5
           :c1 1.0
           :c2 0.7978845608}
 :eqns    [{:op :stablehlo/power    :invars [:x :c_three] :outvars [:t0]}
           {:op :stablehlo/multiply :invars [:t0 0.044715] :outvars [:t1]}
           {:op :stablehlo/add      :invars [:x :t1]      :outvars [:t2]}
           {:op :stablehlo/multiply :invars [:t2 :c2]     :outvars [:t3]}
           {:op :stablehlo/tanh     :invars [:t3]         :outvars [:t4]}
           {:op :stablehlo/add      :invars [:t4 :c1]     :outvars [:t5]}
           {:op :stablehlo/multiply :invars [:x :t5]      :outvars [:t6]}
           {:op :stablehlo/multiply :invars [:t6 :c0]     :outvars [:y]}]}

```

#### Malli Schema Validation (`clj-xla.stablehlo.schema`)

```clojure
(def TensorTypeSchema
  [:vector {:min 2 :max 3}
   [:enum :tensor]
   [:vector :int]
   [:enum :f16 :f32 :bf16 :i32 :i64 :pred]])

(def EquationSchema
  [:map
   [:op keyword?]
   [:invars [:vector [or symbol? keyword? number?]]]
   [:outvars [:vector [or symbol? keyword?]]]
   [:attrs {:optional true} map?]])

(def GraphSchema
  [:map
   [:name string?]
   [:invars [:vector [:tuple symbol? map?]]]
   [:outvars [:vector symbol?]]
   [:eqns [:vector EquationSchema]]])

```

---

### Layer 3: Tracing & Mathematical DSL (`clj-xla.trace` & `clj-xla.tensor`)

Layer 3 allows developers and AI agents to write standard functional math expressions using shadowed Clojure operators.

#### Operator Shadowing & Auto-Lifting (`clj-xla.tensor`)

* Core operators (`+`, `*`, `-`, `/`, `pow`, `tanh`, `sqrt`, `dot-general`, `slice`) are defined in `clj-xla.tensor`.
* When called with raw numerical scalars, values are automatically lifted into constant SSA nodes (`:stablehlo/constant`).
* When executed inside a `trace` context, operations write equations to a thread-local SSA state accumulator.

#### Pure Clojure Kernel Example

```clojure
(ns clj-xla.example.kernels
  (:refer-clojure :exclude [+ * - / min max pow tanh sqrt])
  (:require [clj-xla.tensor :refer [+ * - / min max pow tanh sqrt]]))

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
* Merges forward and backward equations, appending optimizer steps (e.g., AdamW state updates).


2. **Frontend Graph Optimizations:**
* **Dead Code Elimination (DCE):** Prunes unused SSA nodes not connected to `:outvars`.
* **Constant Folding:** Pre-computes purely scalar static subgraphs during tracing.
* **`vmap` Vectorization:** Automatically maps batch dimensions over unbatched single-sample functions.



---

## 4. Target Use Case Architectures

### A. Non-Autoregressive Generation (DiffusionGemma)

* **Strategy:** Compile the heavy backbone model into a static StableHLO executable.
* **Control Loop:** Pure Clojure code orchestrates the 256-token canvas, entropy estimation, and self-conditioning loops, invoking the compiled `PjRtLoadedExecutable` per denoising step on persistent `PjRtBuffer` handles.

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
4. Execute instantly (< 50ms total latency) over existing device memory buffers without losing GPU state.

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
        ├── pjrt.clj           ;; Panama bindings to libpjrt_cuda/cpu
        ├── stablehlo.clj      ;; EDN SSA schema, validation, & MLIR printer
        ├── tensor.clj         ;; Shadowed operators & scalar auto-lifting
        ├── trace.clj          ;; Symbolic tracing engine
        ├── autodiff.clj       ;; Reverse-mode VJP auto-differentiation
        └── safetensors.clj    ;; Zero-copy off-heap weight loader

```

### Development Phases

* **Phase 1: Foundation (PJRT & Memory)**
* Implement Panama bindings for `PjRtClient`, `PjRtBuffer`, `PjRtLoadedExecutable`.
* Build mmap `.safetensors` parser into off-heap `MemorySegment`s.


* **Phase 2: StableHLO IR & Tracing Engine**
* Implement Malli schemas for Jaxpr-style EDN graphs.
* Build `clj-xla.tensor` shadowed operator library and symbolic tracer.
* Write EDN-to-StableHLO MLIR text printer.


* **Phase 3: Autodiff & Optimizations**
* Build reverse-mode VJP generator for standard ops (`dot_general`, `add`, `multiply`, `reduce_mean`).
* Add DCE and constant-folding passes.


* **Phase 4: High-Level Models & Agent Tooling**
* Implement Gemma / FunctionGemma 270M fine-tuning pipelines.
* Build DiffusionGemma discrete text diffusion canvas runtime.
* Package agent schema validation tooling for autonomous graph synthesis.
