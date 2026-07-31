(ns clj-xla.trace-test
  "Unit and generative property tests for clj-xla.trace symbolic graph tracing."
  (:refer-clojure :exclude [+ *])
  (:require [clj-xla.tensor :refer [+ * pow tanh]]
            [clj-xla.trace :refer [trace-graph]]
            [clojure.test :refer [deftest is testing]]
            [clojure.test.check.clojure-test :refer [defspec]]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]))

(defspec prop-tracing-invariants 50
  (prop/for-all [c-val gen/double]
                (let [f (fn [x] (* x c-val))
                      graph (trace-graph "test_trace" [[:x [:tensor [1 32] :f32]]] f)]
                  (and (= "test_trace" (:name graph))
                       (= [[:x [:tensor [1 32] :f32]]] (:invars graph))
                       (seq (:outvars graph))
                       (seq (:eqns graph))))))

(deftest gelu-kernel-trace-test
  (testing "Symbolic tracing of pure Clojure GELU kernel"
    (let [gelu (fn [x]
                 (let [c-sqrt 0.7978845608]
                   (* 0.5 x (+ 1.0 (tanh (* c-sqrt (+ x (* 0.044715 (pow x 3.0)))))))))
          graph (trace-graph "gelu_kernel" [[:x [:tensor [1 128 768] :f32]]] gelu)]
      (is (= "gelu_kernel" (:name graph)))
      (is (pos? (count (:eqns graph)))))))
