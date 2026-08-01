(ns clj-xla.test-runner
  "Main CLI entrypoint for running the complete test suite."
  (:require [clj-xla.autodiff-test]
            [clj-xla.compile-test]
            [clj-xla.download-hf-test]
            [clj-xla.generation-test]
            [clj-xla.kernels-test]
            [clj-xla.models.gemma-test]
            [clj-xla.models.gpt2-test]
            [clj-xla.nn.activations-test]
            [clj-xla.nn.attention-test]
            [clj-xla.nn.loss-test]
            [clj-xla.nn.norm-test]
            [clj-xla.opt-test]
            [clj-xla.pjrt-test]
            [clj-xla.safetensors-test]
            [clj-xla.sampling-test]
            [clj-xla.stablehlo-test]
            [clj-xla.tensor-test]
            [clj-xla.tokenizer-test]
            [clj-xla.trace-test]
            [clojure.test :refer [run-tests]]))

(defn -main
  "Runs all unit and generative tests and exits with code 0 on success, 1 on failure."
  [& _args]
  (let [results (run-tests 'clj-xla.stablehlo-test
                           'clj-xla.pjrt-test
                           'clj-xla.compile-test
                           'clj-xla.tensor-test
                           'clj-xla.trace-test
                           'clj-xla.autodiff-test
                           'clj-xla.opt-test
                           'clj-xla.safetensors-test
                           'clj-xla.kernels-test
                           'clj-xla.nn.activations-test
                           'clj-xla.nn.norm-test
                           'clj-xla.nn.attention-test
                           'clj-xla.nn.loss-test
                           'clj-xla.models.gemma-test
                           'clj-xla.models.gpt2-test
                           'clj-xla.sampling-test
                           'clj-xla.tokenizer-test
                           'clj-xla.generation-test
                           'clj-xla.download-hf-test)
        {:keys [fail error]} results]
    (if (and (zero? fail) (zero? error))
      (do (println "All tests passed successfully.")
          (System/exit 0))
      (do (println (str "Test failures detected: " fail " failures, " error " errors."))
          (System/exit 1)))))
