# clj-xla

High-performance Machine Learning compiler framework and runtime for Clojure targeting **Java 25** and **OpenXLA PJRT C API**.

## Features

- **Pedro Domingos' Declarative Tensor Logic:** Homoiconic Hiccup-style AST DSL (`clj-xla.logic.*`) unifying relational logic, tensor contraction, broadcasting, and neural network layers.
- **Pure EDN SSA Graph IR:** Flat Single Static Assignment (SSA) computation graphs governed by Malli schemas, lowered directly from Tensor Logic ASTs.
- **Java 25 Project Panama FFM:** Zero-copy native bindings to OpenXLA's PJRT C API (`pjrt_c_api.h`) via `java.lang.foreign`.
- **Sub-Millisecond REPL Feedback:** SHA-256 graph hash compilation caching (`clj-xla.compile`) bypassing XLA LLVM codegen on warm REPL evaluations.
- **Multi-Backend OpenXLA Execution:** Seamless hardware execution across CPU, AMD ROCm, Intel SYCL, and NVIDIA CUDA.
- **Pure Clojure LLM Implementations:** Gemma 2, Gemma 3, Gemma 4 (E2B, E4B), SmolLM, and GPT-2 running purely via XLA compilation without manual host matrix math.

---

## Setup & Quickstart

### 1. Requirements
- **Java 25+** (`java --version`)
- **Clojure 1.12+** (`clj --version`)

### 2. Fetch Precompiled PJRT Plugin

Fetch the OpenXLA CPU shared binary plugin into `bin/`:

```bash
clj scripts/fetch_pjrt_binaries.clj cpu
```

### 3. Start Socket REPL

Start a standard Clojure Socket REPL listening on port `5555`:

```bash
clj -M:repl
```

---

## Usage Examples

### 1. Declarative Tensor Logic (Hiccup AST)

Express neural network operations, matrix contractions, and activations in pure declarative Clojure data structures:

```clojure
(ns example.logic
  (:require [clj-xla.core :as xla]
            [clj-xla.logic.lower :as lower]))

;; Initialize runtime context
(def ctx (xla/init-backend! :cpu))

;; Define model signature (invars)
(def invars [[:x [:tensor [1 16 768] :f32]]
             [:w [:tensor [768 768] :f32]]
             [:gamma [:tensor [768] :f32]]])

;; Declare computation using Tensor Logic AST
(def ast
  [[:rms-norm [:h_norm :b :p :d] [:x :b :p :d] [:gamma :d] {:eps 1e-6}]
   [:matmul [:out :b :p :d] [:h_norm :b :p :din] [:w :din :d]]
   [:gelu [:y :b :p :d] [:out :b :p :d]]])

;; Lower AST to StableHLO SSA graph and compile
(def graph (lower/ast->graph "dense_block" invars ast #{:y}))
(def exec (xla/compile-graph ctx graph))
```

### 2. Direct EDN SSA Graph

Low-level homoiconic StableHLO graph construction:

```clojure
(ns example.ssa
  (:require [clj-xla.core :as xla]))

(def ctx (xla/init-backend! :cpu))

(def graph
  {:name "scaled_gelu"
   :invars [[:x [:tensor [1 128 768] :f32]]]
   :outvars [:y]
   :eqns [{:op :stablehlo/constant :value 0.5 :outvars [:c0]}
          {:op :stablehlo/multiply :invars [:x :c0] :outvars [:y]}]})

(def exec (xla/compile-graph ctx graph))
```

---

## Running Models

CLI scripts are included for running end-to-end autoregressive generation:

```bash
# GPT-2
clj -M scripts/gpt2_inference.clj --prompt "The capital of France is"

# SmolLM-135M
clj -M scripts/smollm_inference.clj --prompt "In a galaxy far away"

# Gemma 3
clj -M scripts/gemma3_inference.clj --prompt "Explain quantum computing" --backend cpu

# Gemma 4
clj -M scripts/gemma4_inference.clj --prompt "Write a short poem" --backend cpu
```

---

## Architecture

See [DESIGN.md](DESIGN.md) for detailed layer specifications and [docs/](docs/) for model specifications, wiki, and hardware notes.
