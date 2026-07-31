# clj-xla

High-performance Machine Learning compiler framework and runtime for Clojure targeting **Java 25** and **OpenXLA PJRT C API**.

## Features

- **Pure EDN SSA Graph IR:** Homoiconic, flat Single Static Assignment (SSA) computation graphs governed by Malli schemas.
- **Java 25 Project Panama FFM:** Zero-copy native bindings to OpenXLA's PJRT C API (`pjrt_c_api.h`) via `java.lang.foreign`.
- **Sub-Millisecond REPL Feedback:** SHA-256 graph hash compilation caching (`clj-xla.compile`) bypassing XLA LLVM codegen on warm REPL evaluations.
- **Zero Submodules:** Automated precompiled PJRT binary fetcher (`scripts/fetch_pjrt_binaries.clj`).

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

## Usage Example

Connect to the Socket REPL (`localhost:5555`) or evaluate from your Clojure editor:

```clojure
(ns example
  (:require [clj-xla.core :as xla]))

;; Initialize CPU runtime context
(xla/init-cpu!)

;; Define a flat EDN SSA graph
(def graph
  {:name "gelu_block"
   :invars [[:x [:tensor [1 128 768] :f32]]]
   :outvars [:y]
   :eqns [{:op :stablehlo/constant :value 0.5 :outvars [:c0]}
          {:op :stablehlo/multiply :invars [:x :c0] :outvars [:y]}]})

;; Compile graph to PjRtLoadedExecutable (cached in-memory)
(def exec (xla/compile-graph graph))
```

---

## Architecture

See [DESIGN.md](DESIGN.md) for detailed layer specifications and [XLA.md](XLA.md) for vendor artifact management details.
