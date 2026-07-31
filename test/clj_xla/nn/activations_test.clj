(ns clj-xla.nn.activations-test
  "Unit and generative tests for neural network activation functions."
  (:require [clj-xla.nn.activations :as act]
            [clj-xla.tensor :as t :refer [tracer?]]
            [clj-xla.trace :refer [trace-graph]]
            [clojure.test :refer [deftest is testing]]
            [clojure.test.check.clojure-test :refer [defspec]]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]))

(defspec prop-activations-tracing 50
  (prop/for-all [_seq-len (gen/choose 1 64)]
                (let [g-gelu (trace-graph "gelu_test" [[:x [:tensor [1 128 768] :f32]]] act/gelu)
                      g-relu (trace-graph "relu_test" [[:x [:tensor [1 128 768] :f32]]] act/relu)]
                  (and (= "gelu_test" (:name g-gelu))
                       (= "relu_test" (:name g-relu))
                       (seq (:eqns g-gelu))
                       (seq (:eqns g-relu))))))

(deftest activation-tracer-test
  (testing "Activation functions return Tracer on Tracer input"
    (let [x (t/->Tracer :x [:tensor [2 4] :f32])]
      (is (tracer? (act/gelu x)))
      (is (tracer? (act/silu x)))
      (is (tracer? (act/relu x))))))
