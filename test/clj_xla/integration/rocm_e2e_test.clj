(ns clj-xla.integration.rocm-e2e-test
  "End-to-end integration test suite for AMD ROCm GPUs, multi-device topology, and CPU vs ROCm execution parity."
  (:require [clj-xla.pjrt :as pjrt]
            [clj-xla.pjrt.version :as v]
            [clj-xla.stablehlo :as shlo]
            [clj-xla.test.parity :as p]
            [clojure.java.io :as io]
            [clojure.test :refer [deftest is testing]]))

(deftest test-rocm-gpu-integration
  (let [probe (v/probe-system-driver)
        rocm-so (io/file "bin/libpjrt_rocm.so")]
    (if (and (contains? (:detected-backends probe) :rocm) (.exists rocm-so))
      (testing "AMD ROCm GPU initialization, device discovery, and hardware compilation"
        (let [api (pjrt/load-plugin! "bin/libpjrt_rocm.so")]
          (is (some? (:api-ptr api)))
          (let [client (pjrt/create-client api)
                devs (pjrt/addressable-devices api client)]
            (is (some? client))
            (is (pos? (count devs)) "At least 1 ROCm device handle must be addressable")
            (when (>= (count devs) 2)
              (println (format "Dual AMD GPU Topology Detected: %d addressable ROCm devices." (count devs))))
            (let [graph {:name "rocm_e2e_module"
                         :invars [[:x [:tensor [1 64] :f32]]
                                  [:y [:tensor [1 64] :f32]]]
                         :outvars [:z]
                         :eqns [{:op :stablehlo/multiply :invars [:x :y] :outvars [:z]}]}
                  mlir (shlo/graph->mlir-text graph)
                  exec (pjrt/compile-mlir api client mlir)]
              (is (some? exec) "ROCm hardware binary compilation must succeed")))))
      (println "Skipping ROCm E2E hardware test (ROCm backend or bin/libpjrt_rocm.so not loaded)."))))

(deftest test-cpu-vs-rocm-execution-parity
  (let [probe (v/probe-system-driver)
        rocm-so (io/file "bin/libpjrt_rocm.so")
        cpu-so (io/file "bin/libpjrt_cpu.so")]
    (when (and (contains? (:detected-backends probe) :rocm) (.exists rocm-so) (.exists cpu-so))
      (testing "CPU vs ROCm execution parity on AMD GPU hardware"
        (let [api-cpu (pjrt/load-plugin! "bin/libpjrt_cpu.so")
              cli-cpu (pjrt/create-client api-cpu)
              api-rocm (pjrt/load-plugin! "bin/libpjrt_rocm.so")
              cli-rocm (pjrt/create-client api-rocm)
              graph {:name "parity_test_graph"
                     :invars [[:x [:tensor [1 16] :f32]]]
                     :outvars [:y]
                     :eqns [{:op :stablehlo/constant :value 2.5 :outvars [:c0]}
                            {:op :stablehlo/multiply :invars [:x :c0] :outvars [:y]}]}
              inputs {:x (float-array (range 16))}
              res (p/compare-cpu-vs-device api-cpu cli-cpu api-rocm cli-rocm graph inputs {:atol 1e-4 :rtol 1e-4})]
          (is (true? (:match? res)) (str "CPU vs ROCm execution mismatch: " res))
          (is (< (:max-diff res) 1e-4)))))))
