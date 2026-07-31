* **Rule 1: Strict TDD.** Write generative tests (`clojure.test.check`) for invariants *before* implementing core logic.
* **Rule 2: Pure Functions.** The core must remain pure (Sans-IO). Side effects are strictly isolated to boundary shells.
* **Rule 3: Clean CI.** PRs must run `clojure -M:format`, `clojure -M:lint`, and tests (`clojure -M:test -m clj-xla.test-runner`) successfully before submission. Do not ignore linter warnings.
