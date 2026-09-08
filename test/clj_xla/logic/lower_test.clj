(ns clj-xla.logic.lower-test
  "Generative property and unit tests for Tensor Logic StableHLO lowering."
  (:require [clj-xla.core :as xla]
            [clj-xla.logic.lower :as lower]
            [clj-xla.stablehlo :as shlo]
            [clojure.test :refer [deftest is]]
            [clojure.test.check.clojure-test :refer [defspec]]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]))

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

(defspec prop-lower-dynamic-slice-produces-valid-graph
  50
  (prop/for-all [b (gen/choose 1 4)
                 s (gen/choose 16 128)
                 d (gen/choose 32 256)]
                (let [invars [[:x [:tensor [b s d] :f32]]
                              [:pos [:tensor [1] :i32]]]
                      ast [:dynamic-slice [:y :b :one :d] [:x :b :p :d]
                           {:slice-sizes [1 1 d] :start-indices [0 :pos 0]}]
                      graph (lower/ast->graph "ds_graph" invars ast #{:y})]
                  (and (shlo/validate-graph graph)
                       (= [:y] (:outvars graph))
                       (boolean (some #(= :stablehlo/dynamic_slice (:op %)) (:eqns graph)))))))

(deftest test-end-to-end-dynamic-slice-execution
  (let [ctx (xla/get-context)
        invars [[:x [:tensor [1 10 4] :f32]]
                [:pos [:tensor [1] :i32]]]
        ast [:dynamic-slice [:y :b :one :d] [:x :b :p :d]
             {:slice-sizes [1 1 4] :start-indices [0 :pos 0]}]
        graph (lower/ast->graph "e2e_ds" invars ast #{:y})
        compiled (xla/compile-graph ctx graph)
        x-data (float-array (range 40))
        pos-data (int-array [3])
        out-buf (xla/execute compiled x-data pos-data)
        res (xla/to-host-slice out-buf 0 4 4)]
    ;; At pos=3, elements for slice [1 1 4] starting at [0 3 0] are [12.0, 13.0, 14.0, 15.0]
    (is (= 12.0 (nth res 0)))
    (is (= 13.0 (nth res 1)))
    (is (= 14.0 (nth res 2)))
    (is (= 15.0 (nth res 3)))
    (xla/destroy-buffer! out-buf)))

(defspec prop-lower-while-produces-valid-graph
  50
  (prop/for-all [_lim (gen/choose 1 100)]
                (let [invars [[:init [:tensor [1] :i32]]]
                      ast [:while [:final_step] [:init]
                           {:cond-mlir "    ^bb0(%s: tensor<1xi32>):\n      %c = stablehlo.constant dense<10> : tensor<1xi32>\n      %cmp = \"stablehlo.compare\"(%s, %c) {comparison_direction = #stablehlo<comparison_direction LT>} : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>\n      %res = stablehlo.reshape %cmp : (tensor<1xi1>) -> tensor<i1>\n      \"stablehlo.return\"(%res) : (tensor<i1>) -> ()"
                            :body-mlir "    ^bb0(%s: tensor<1xi32>):\n      %one = stablehlo.constant dense<1> : tensor<1xi32>\n      %next = stablehlo.add %s, %one : tensor<1xi32>\n      \"stablehlo.return\"(%next) : (tensor<1xi32>) -> ()"}]
                      graph (lower/ast->graph "while_graph" invars ast #{:final_step})]
                  (and (shlo/validate-graph graph)
                       (= [:final_step] (:outvars graph))
                       (boolean (some #(= :stablehlo/while (:op %)) (:eqns graph)))))))

(deftest test-end-to-end-while-execution
  (let [ctx (xla/get-context)
        invars [[:init [:tensor [1] :i32]]]
        ast [:while [:final_step] [:init]
             {:cond-mlir "    ^bb0(%s: tensor<1xi32>):\n      %c = stablehlo.constant dense<5> : tensor<1xi32>\n      %cmp = \"stablehlo.compare\"(%s, %c) {comparison_direction = #stablehlo<comparison_direction LT>} : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>\n      %res = stablehlo.reshape %cmp : (tensor<1xi1>) -> tensor<i1>\n      \"stablehlo.return\"(%res) : (tensor<i1>) -> ()"
              :body-mlir "    ^bb0(%s: tensor<1xi32>):\n      %one = stablehlo.constant dense<1> : tensor<1xi32>\n      %next = stablehlo.add %s, %one : tensor<1xi32>\n      \"stablehlo.return\"(%next) : (tensor<1xi32>) -> ()"}]
        graph (lower/ast->graph "e2e_while" invars ast #{:final_step})
        compiled (xla/compile-graph ctx graph)
        init-data (int-array [0])
        out-buf (xla/execute compiled init-data)
        res (xla/to-host-slice out-buf 0 1 4)
        val (Float/floatToIntBits (first res))]
    (is (= 5 val))
    (xla/destroy-buffer! out-buf)))

(defspec prop-lower-while-with-cond-ast-produces-valid-graph
  50
  (prop/for-all [_lim (gen/choose 1 100)]
                (let [invars [[:init [:tensor [] :i32]]
                              [:max_step [:tensor [] :i32]]
                              [:false_c [:tensor [] :i1]]]
                      cond-ast [:cond [:cond_out] {:args [:cur_step :target_max :cur_stopped]}
                                [:compare [:step_lt] [:cur_step] [:target_max] {:direction "LT"}]
                                [:not [:not_stopped] [:cur_stopped]]
                                [:and [:cond_out] [:step_lt] [:not_stopped]]]
                      loop-ast [:while [:final_step :final_max :final_stopped]
                                [:init :max_step :false_c]
                                {:body-mlir "    ^bb0(%s: tensor<i32>, %m: tensor<i32>, %st: tensor<i1>):\n      \"stablehlo.return\"(%s, %m, %st) : (tensor<i32>, tensor<i32>, tensor<i1>) -> ()"}
                                cond-ast]
                      graph (lower/ast->graph "while_cond_graph" invars loop-ast [:final_step])]
                  (and (shlo/validate-graph graph)
                       (= [:final_step] (:outvars graph))
                       (boolean (some #(= :stablehlo/while (:op %)) (:eqns graph)))))))

(deftest test-end-to-end-while-cond-ast-execution
  (let [ctx (xla/get-context)
        invars [[:init [:tensor [] :i32]]
                [:max_step [:tensor [] :i32]]
                [:false_c [:tensor [] :i1]]]
        cond-ast [:cond [:cond_out] {:args [:cur_step :target_max :cur_stopped]}
                  [:compare [:step_lt] [:cur_step] [:target_max] {:direction "LT"}]
                  [:not [:not_stopped] [:cur_stopped]]
                  [:and [:cond_out] [:step_lt] [:not_stopped]]]
        loop-ast [:while [:final_step :final_max :final_stopped]
                  [:init :max_step :false_c]
                  {:body-mlir "    ^bb0(%s: tensor<i32>, %m: tensor<i32>, %st: tensor<i1>):\n      %one = stablehlo.constant dense<1> : tensor<i32>\n      %next = stablehlo.add %s, %one : tensor<i32>\n      \"stablehlo.return\"(%next, %m, %st) : (tensor<i32>, tensor<i32>, tensor<i1>) -> ()"}
                  cond-ast]
        graph (lower/ast->graph "e2e_while_cond" invars loop-ast [:final_step])
        compiled (xla/compile-graph ctx graph)
        init-data (int-array [0])
        max-data (int-array [7])
        false-data (byte-array [0])
        out-buf (xla/execute compiled init-data max-data false-data)
        res (xla/to-host-slice out-buf 0 1 4)
        val (Float/floatToIntBits (first res))]
    (is (= 7 val))
    (xla/destroy-buffer! out-buf)))

(defspec prop-lower-argmax-produces-valid-graph
  50
  (prop/for-all [b (gen/choose 1 4)
                 v (gen/choose 16 128)]
                (let [invars [[:logits [:tensor [b v] :f32]]]
                      ast [:argmax [:out] [:logits] {:axis 1}]
                      graph (lower/ast->graph "argmax_graph" invars ast #{:out})]
                  (and (shlo/validate-graph graph)
                       (= [:out] (:outvars graph))
                       (boolean (some #(= :stablehlo/argmax (:op %)) (:eqns graph)))))))

(deftest test-end-to-end-argmax-execution
  (let [ctx (xla/get-context)
        invars [[:logits [:tensor [2 4] :f32]]]
        ast [:argmax [:out] [:logits] {:axis 1}]
        graph (lower/ast->graph "e2e_argmax" invars ast #{:out})
        compiled (xla/compile-graph ctx graph)
        logits-data (float-array [1.0 9.0 2.0 3.0
                                  0.5 1.5 8.5 2.5])
        out-buf (xla/execute compiled logits-data)
        res (xla/to-host-slice out-buf 0 2 8)
        idx0 (Float/floatToIntBits (nth res 0))
        idx1 (Float/floatToIntBits (nth res 1))]
    (is (= 1 idx0))
    (is (= 2 idx1))
    (xla/destroy-buffer! out-buf)))

(defspec prop-lower-while-with-body-ast-produces-valid-graph
  50
  (prop/for-all [_lim (gen/choose 1 100)]
                (let [invars [[:init [:tensor [] :i32]]
                              [:max_step [:tensor [] :i32]]
                              [:false_c [:tensor [] :i1]]]
                      cond-ast [:cond [:cond_out] {:args [:cur_step :target_max :cur_stopped]}
                                [:compare [:step_lt] [:cur_step] [:target_max] {:direction "LT"}]
                                [:not [:not_stopped] [:cur_stopped]]
                                [:and [:cond_out] [:step_lt] [:not_stopped]]]
                      body-ast [:body [:next_step :target_max :cur_stopped]
                                {:args [:cur_step :target_max :cur_stopped]}
                                [:constant [:c_one] {:value 1 :type [:tensor [] :i32] :shape []}]
                                [:+ [:next_step] [:cur_step] [:c_one]]]
                      loop-ast [:while [:final_step :final_max :final_stopped]
                                [:init :max_step :false_c]
                                cond-ast
                                body-ast]
                      graph (lower/ast->graph "while_body_graph" invars loop-ast [:final_step])]
                  (and (shlo/validate-graph graph)
                       (= [:final_step] (:outvars graph))
                       (boolean (some #(= :stablehlo/while (:op %)) (:eqns graph)))))))

(deftest test-end-to-end-while-declarative-body-execution
  (let [ctx (xla/get-context)
        invars [[:init [:tensor [] :i32]]
                [:max_step [:tensor [] :i32]]
                [:false_c [:tensor [] :i1]]]
        cond-ast [:cond [:cond_out] {:args [:cur_step :target_max :cur_stopped]}
                  [:compare [:step_lt] [:cur_step] [:target_max] {:direction "LT"}]
                  [:not [:not_stopped] [:cur_stopped]]
                  [:and [:cond_out] [:step_lt] [:not_stopped]]]
        body-ast [:body [:next_step :target_max :cur_stopped]
                  {:args [:cur_step :target_max :cur_stopped]}
                  [:constant [:c_one] {:value 1 :type [:tensor [] :i32] :shape []}]
                  [:+ [:next_step] [:cur_step] [:c_one]]]
        loop-ast [:while [:final_step :final_max :final_stopped]
                  [:init :max_step :false_c]
                  cond-ast
                  body-ast]
        graph (lower/ast->graph "e2e_while_body" invars loop-ast [:final_step])
        compiled (xla/compile-graph ctx graph)
        init-data (int-array [0])
        max-data (int-array [10])
        false-data (byte-array [0])
        out-buf (xla/execute compiled init-data max-data false-data)
        res (xla/to-host-slice out-buf 0 1 4)
        val (Float/floatToIntBits (first res))]
    (is (= 10 val))
    (xla/destroy-buffer! out-buf)))

(defspec prop-lower-dynamic-update-slice-produces-valid-graph
  50
  (prop/for-all [b (gen/choose 1 2)
                 s (gen/choose 8 32)
                 d (gen/choose 16 64)]
                (let [invars [[:cache [:tensor [b s d] :f32]]
                              [:new_slice [:tensor [b 1 d] :f32]]
                              [:pos [:tensor [1] :i32]]]
                      ast [:dynamic-update-slice [:out :b :s :d] [:cache :b :s :d] [:new_slice :b :one :d]
                           {:start-indices [0 :pos 0]}]
                      graph (lower/ast->graph "dus_graph" invars ast #{:out})]
                  (and (shlo/validate-graph graph)
                       (= [:out] (:outvars graph))
                       (boolean (some #(= :stablehlo/dynamic_update_slice (:op %)) (:eqns graph)))))))

(deftest test-end-to-end-dynamic-update-slice-execution
  (let [ctx (xla/get-context)
        invars [[:cache [:tensor [1 4 4] :f32]]
                [:new_slice [:tensor [1 1 4] :f32]]
                [:pos [:tensor [1] :i32]]]
        ast [:dynamic-update-slice [:out :b :s :d] [:cache :b :s :d] [:new_slice :b :one :d]
             {:start-indices [0 :pos 0]}]
        graph (lower/ast->graph "e2e_dus" invars ast #{:out})
        compiled (xla/compile-graph ctx graph)
        cache-data (float-array (repeat 16 0.0))
        slice-data (float-array [9.0 8.0 7.0 6.0])
        pos-data (int-array [2])
        out-buf (xla/execute compiled cache-data slice-data pos-data)
        res (xla/to-host-slice out-buf 0 16 16)]
    ;; Position 2 row in [1 4 4] should be [9.0 8.0 7.0 6.0] (indices 8, 9, 10, 11)
    (is (= 0.0 (nth res 0)))
    (is (= 9.0 (nth res 8)))
    (is (= 8.0 (nth res 9)))
    (is (= 7.0 (nth res 10)))
    (is (= 6.0 (nth res 11)))
    (is (= 0.0 (nth res 12)))
    (xla/destroy-buffer! out-buf)))

