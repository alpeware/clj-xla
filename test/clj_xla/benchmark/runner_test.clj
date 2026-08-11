(ns clj-xla.benchmark.runner-test
  "Unit & generative property tests for PJRT backend benchmark runner execution harness."
  (:require [clj-xla.benchmark.runner :as runner]
            [clj-xla.benchmark.workloads :as bw]
            [clj-xla.core :as xla]
            [clojure.test :refer [deftest is testing]]))

(deftest test-benchmark-single-workload-cpu
  (testing "Running a single GEMM benchmark workload on CPU backend"
    (let [ctx (xla/init-cpu!)
          wl (:gemm-fp32 bw/WORKLOADS)
          res (runner/benchmark-workload ctx wl {:warmup-iters 2 :measure-iters 3})]
      (is (map? res))
      (is (= :gemm-fp32 (:kernel res)))
      (is (= :cpu (:backend res)))
      (is (pos? (:cold-ms res)))
      (is (pos? (:mean-ms res)))
      (is (pos? (:p50-ms res)))
      (is (pos? (:p99-ms res)))
      (is (pos? (:tflops res))))))

(deftest test-run-backend-benchmarks-cpu
  (testing "Running full workload benchmark suite on CPU backend"
    (let [results (runner/run-backend-benchmarks :cpu {:warmup-iters 1 :measure-iters 2})]
      (is (vector? results))
      (is (= (count bw/WORKLOADS) (count results)))
      (doseq [res results]
        (is (= :cpu (:backend res)))
        (is (pos? (:mean-ms res)))))))
