(ns clj-xla.compile-test
  "Unit tests for compilation caching."
  (:require [clj-xla.compile :as compile]
            [clj-xla.pjrt :as pjrt]
            [clojure.test :refer [deftest is testing]]))

(deftest compilation-caching-test
  (testing "SHA-256 graph compilation caching"
    (compile/clear-cache!)
    (let [api (pjrt/load-plugin! "bin/libpjrt_cpu.so")
          client (pjrt/create-client api)
          graph {:name "cached_graph"
                 :invars [[:x [:tensor [2 32] :f32]]]
                 :outvars [:y]
                 :eqns [{:op :stablehlo/constant :value 3.0 :outvars [:c0]}
                        {:op :stablehlo/multiply :invars [:x :c0] :outvars [:y]}]}
          exec1 (compile/compile-graph api client graph)
          exec2 (compile/compile-graph api client graph)]
      (is (some? exec1))
      (is (= exec1 exec2) "Cached compilation must return identical executable handle")
      (compile/clear-cache!))))
