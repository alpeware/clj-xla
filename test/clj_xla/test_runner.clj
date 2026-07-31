(ns clj-xla.test-runner
  "Main CLI entrypoint for running the test suite."
  (:require [clj-xla.compile-test]
            [clj-xla.pjrt-test]
            [clj-xla.stablehlo-test]
            [clojure.test :refer [run-tests]]))

(defn -main
  "Runs all unit and generative tests and exits with code 0 on success, 1 on failure."
  [& _args]
  (let [results (run-tests 'clj-xla.stablehlo-test
                           'clj-xla.pjrt-test
                           'clj-xla.compile-test)
        {:keys [fail error]} results]
    (if (and (zero? fail) (zero? error))
      (do (println "All tests passed successfully.")
          (System/exit 0))
      (do (println (str "Test failures detected: " fail " failures, " error " errors."))
          (System/exit 1)))))
