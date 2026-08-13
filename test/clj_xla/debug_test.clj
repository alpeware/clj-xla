(ns clj-xla.debug-test
  "Generative and unit tests for clj-xla.debug module (XLA metadata, NaN checkify, and tracing assertions)."
  (:require [clj-xla.debug :as debug]
            [clj-xla.stablehlo :as shlo]
            [clj-xla.tensor :as t]
            [clj-xla.trace :as trace]
            [clojure.string :as str]
            [clojure.test :refer [deftest is testing]]
            [clojure.test.check.clojure-test :refer [defspec]]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]))

(deftest test-xla-metadata-annotation
  (testing "Attaching XLA metadata to traced graph equations"
    (let [graph (trace/trace-graph "test_meta" [[:x [:tensor [128 128] :f32]]]
                                   (fn [x]
                                     (debug/with-xla-metadata {:op-name "gemma/layer_0/rms_norm"
                                                               :source-file "attention.clj"
                                                               :source-line 42}
                                       (t/+ x 1.0))))
          mlir-text (shlo/graph->mlir-text graph)]
      (is (map? graph))
      (is (string? mlir-text))
      (is (str/includes? mlir-text "gemma/layer_0/rms_norm")))))

(deftest test-checkify-nan-instrumentation
  (testing "Instrumenting tensor graph with checkify NaN assertion"
    (let [graph (trace/trace-graph "test_nan" [[:x [:tensor [128 128] :f32]]]
                                   (fn [x]
                                     (let [y (t/* x 2.0)
                                           _ (debug/check-non-nan y "NaN detected in intermediate product")]
                                       y)))]
      (is (map? graph))
      (is (some (fn [eqn] (= (:op eqn) :debug/check-non-nan)) (:eqns graph))))))

(defspec prop-xla-metadata-scoping-invariant
  50
  (prop/for-all [op-name (gen/not-empty gen/string-alphanumeric)]
                (let [graph (trace/trace-graph "test_prop" [[:x [:tensor [128 128] :f32]]]
                                               (fn [x]
                                                 (debug/with-xla-metadata {:op-name op-name}
                                                   (t/+ x 1.0))))
                      eqns (:eqns graph)]
                  (every? (fn [eqn]
                            (= (get-in eqn [:metadata :op-name]) op-name))
                          eqns))))
