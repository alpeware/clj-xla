(ns clj-xla.logic.lower-test
  "Generative property and unit tests for Tensor Logic StableHLO lowering."
  (:require [clj-xla.core :as xla]
            [clj-xla.logic.lower :as lower]
            [clj-xla.stablehlo :as shlo]
            [clojure.test :refer [deftest is]]))

(deftest test-lower-gemm-produces-valid-ssa-graph
  (let [invars [[:x [:tensor [2 16 32] :f32]]
                [:w [:tensor [32 64] :f32]]]
        ast [:= [:y :b :m :n] [:x :b :m :k] [:w :k :n]]
        graph (lower/ast->graph "gemm_graph" invars ast #{:y})]
    (is (shlo/validate-graph graph))
    (is (= [:y] (:outvars graph)))
    (let [ops (mapv :op (:eqns graph))]
      (is (some #(= :stablehlo/dot_general %) ops)))))

(deftest test-lower-implicit-accumulation
  (let [invars [[:x [:tensor [2 16 32] :f32]]
                [:y [:tensor [2 16 32] :f32]]]
        ast [:block {}
             [:= [:out :b :m :d] [:x :b :m :d]]
             [:= [:out :b :m :d] [:y :b :m :d]]]
        graph (lower/ast->graph "accum_graph" invars ast #{:out})]
    (is (shlo/validate-graph graph))
    (let [ops (mapv :op (:eqns graph))]
      ;; Implicit accumulation emits an addition equation
      (is (some #(= :stablehlo/add %) ops)))))

(deftest test-lower-post-activation-scale
  (let [invars [[:x [:tensor [2 16 32] :f32]]
                [:w [:tensor [32 64] :f32]]]
        ast [:= [:y :b :m :n] {:scale 0.5} [:x :b :m :k] [:w :k :n]]
        graph (lower/ast->graph "scaled_gemm" invars ast #{:y})]
    (is (shlo/validate-graph graph))
    (let [ops (mapv :op (:eqns graph))]
      (is (some #(= :stablehlo/multiply %) ops)))))

(deftest test-end-to-end-pjrt-execution-parity
  (let [ctx (xla/get-context)
        invars [[:x [:tensor [2 4] :f32]]
                [:w [:tensor [4 2] :f32]]]
        ast [:= [:y :m :n] [:x :m :k] [:w :k :n]]
        graph (lower/ast->graph "e2e_gemm" invars ast #{:y})
        compiled (xla/compile-graph ctx graph)
        x-data (float-array [1.0 2.0 3.0 4.0
                             5.0 6.0 7.0 8.0])
        w-data (float-array [1.0 0.0
                             0.0 1.0
                             1.0 0.0
                             0.0 1.0])
        out-buf (xla/execute compiled x-data w-data)
        res (xla/to-host-slice out-buf 0 4 4)]
    ;; Row 0: [1+3, 2+4] = [4.0, 6.0]
    ;; Row 1: [5+7, 6+8] = [12.0, 14.0]
    (is (= 4.0 (nth res 0)))
    (is (= 6.0 (nth res 1)))
    (is (= 12.0 (nth res 2)))
    (is (= 14.0 (nth res 3)))
    (xla/destroy-buffer! out-buf)))
