(ns clj-xla.test-runner
  "Main CLI entrypoint for running the complete test suite."
  (:require [clj-xla.autodiff-test]
            [clj-xla.compile-test]
            [clj-xla.core-test]
            [clj-xla.download-hf-test]
            [clj-xla.fetch-pjrt-binaries-test]
            [clj-xla.generation-test]
            [clj-xla.integration.rocm-e2e-test]
            [clj-xla.kernels-test]
            [clj-xla.models.gemma-test]
            [clj-xla.models.gpt2-test]
            [clj-xla.nn.activations-test]
            [clj-xla.nn.attention-test]
            [clj-xla.nn.loss-test]
            [clj-xla.nn.norm-test]
            [clj-xla.opt-test]
            [clj-xla.pjrt-test]
            [clj-xla.pjrt.version-test]
            [clj-xla.safetensors-test]
            [clj-xla.sampling-test]
            [clj-xla.stablehlo-test]
            [clj-xla.tensor-test]
            [clj-xla.test.generators-test]
            [clj-xla.test.isolated-runner :as isolated-runner]
            [clj-xla.test.isolated-runner-test]
            [clj-xla.test.parity-test]
            [clj-xla.test.telemetry :as telemetry]
            [clj-xla.test.telemetry-test]
            [clj-xla.tokenizer-test]
            [clj-xla.trace-test]
            [clojure.test :refer [run-tests]]))

(defn -main
  "Runs all unit, generative, and hardware integration tests, outputting an EDN telemetry report."
  [& _args]
  (let [results (run-tests 'clj-xla.stablehlo-test
                           'clj-xla.pjrt-test
                           'clj-xla.pjrt.version-test
                           'clj-xla.test.generators-test
                           'clj-xla.test.isolated-runner-test
                           'clj-xla.test.parity-test
                           'clj-xla.test.telemetry-test
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
                           'clj-xla.download-hf-test
                           'clj-xla.fetch-pjrt-binaries-test
                           'clj-xla.core-test)
        rocm-res (isolated-runner/run-isolated-test 'clj-xla.integration.rocm-e2e-test {"HIP_VISIBLE_DEVICES" "0" "ROCR_VISIBLE_DEVICES" "0"})
        {:keys [fail error]} results
        rocm-ok? (= (:status rocm-res) :pass)
        _ (telemetry/generate-edn-report {:in-process-results results :isolated-hardware-results rocm-res})]
    (if (and (zero? fail) (zero? error) rocm-ok?)
      (do (println "All unit, generative, and hardware integration tests passed successfully.")
          (System/exit 0))
      (do (println (str "Test failures detected: " fail " failures, " error " errors, ROCm status: " (:status rocm-res)))
          (System/exit 1)))))
