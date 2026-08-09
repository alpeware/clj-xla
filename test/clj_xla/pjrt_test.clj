(ns clj-xla.pjrt-test
  "Unit tests for PJRT plugin loading and compilation."
  (:require [clj-xla.pjrt :as pjrt]
            [clj-xla.stablehlo :as shlo]
            [clojure.test :refer [deftest is testing]]))

(deftest pjrt-cpu-plugin-test
  (testing "Loading CPU PJRT plugin and initializing client"
    (let [api (pjrt/load-plugin! "bin/libpjrt_cpu.so")]
      (is (some? (:api-ptr api)))
      (is (some? (:linker api)))
      (let [[major minor] (pjrt/api-version api)]
        (is (integer? major))
        (is (integer? minor))
        (is (>= major 0))
        (is (>= minor 0)))
      (let [client (pjrt/create-client api)]
        (is (some? client))
        (is (= "cpu" (pjrt/platform-name api client)))))))

(deftest pjrt-compile-mlir-test
  (testing "Compiling StableHLO MLIR text to hardware executable via PJRT"
    (let [api (pjrt/load-plugin! "bin/libpjrt_cpu.so")
          client (pjrt/create-client api)
          graph {:name "test_compile_module"
                 :invars [[:x [:tensor [1 64] :f32]]]
                 :outvars [:y]
                 :eqns [{:op :stablehlo/constant :value 1.5 :outvars [:c0]}
                        {:op :stablehlo/multiply :invars [:x :c0] :outvars [:y]}]}
          mlir (shlo/graph->mlir-text graph)
          exec (pjrt/compile-mlir api client mlir)]
      (is (some? exec)))))
