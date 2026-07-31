# Repository Rules & Coding Guidelines for AI Agents

* **Rule 1: Strict TDD.** Write generative tests (`clojure.test.check`) for invariants *before* implementing core logic.
* **Rule 2: Pure Functions.** The core must remain pure (Sans-IO). Side effects are strictly isolated to boundary shells.
* **Rule 3: Clean CI.** PRs must run `clojure -M:format`, `clojure -M:lint`, and tests (`clojure -M:test -m clj-xla.test-runner`) successfully before submission. Do not ignore linter warnings.
* **Rule 4: Zero Java Escape Hatches (Pure XLA Execution).** All tensor math, neural network layers, and full model forward passes MUST be written in pure Clojure using `clj-xla.tensor` / `clj-xla.trace` and compiled into StableHLO MLIR for OpenXLA execution via PJRT. Under NO circumstances should custom `.java` classes, host-side primitive float array loops (`float[][]`), or manual CPU matrix math engines be created to bypass XLA compilation.
