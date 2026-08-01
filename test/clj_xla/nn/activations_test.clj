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
                      g-silu (trace-graph "silu_test" [[:x [:tensor [1 128 768] :f32]]] act/silu)
                      g-relu (trace-graph "relu_test" [[:x [:tensor [1 128 768] :f32]]] act/relu)
                      g-swiglu (trace-graph "swiglu_test"
                                            [[:x [:tensor [1 16 576] :f32]]
                                             [:gw [:tensor [576 1536] :f32]]
                                             [:uw [:tensor [576 1536] :f32]]
                                             [:dw [:tensor [1536 576] :f32]]]
                                            (fn [x gw uw dw] (act/swiglu x gw uw dw)))
                      g-geglu (trace-graph "geglu_test"
                                           [[:x [:tensor [1 16 2048] :f32]]
                                            [:gw [:tensor [2048 16384] :f32]]
                                            [:uw [:tensor [2048 16384] :f32]]
                                            [:dw [:tensor [16384 2048] :f32]]]
                                           (fn [x gw uw dw] (act/geglu x gw uw dw)))]
                  (and (= "gelu_test" (:name g-gelu))
                       (= "silu_test" (:name g-silu))
                       (= "relu_test" (:name g-relu))
                       (= "swiglu_test" (:name g-swiglu))
                       (= "geglu_test" (:name g-geglu))
                       (seq (:eqns g-gelu))
                       (seq (:eqns g-silu))
                       (seq (:eqns g-swiglu))
                       (seq (:eqns g-geglu))))))

(deftest activation-tracer-test
  (testing "Activation functions return Tracer on Tracer input"
    (let [x (t/->Tracer :x [:tensor [2 4] :f32])
          x-m (t/->Tracer :xm [:tensor [1 16 2048] :f32])
          gw (t/->Tracer :gw [:tensor [2048 16384] :f32])
          uw (t/->Tracer :uw [:tensor [2048 16384] :f32])
          dw (t/->Tracer :dw [:tensor [16384 2048] :f32])]
      (is (tracer? (act/gelu x)))
      (is (tracer? (act/silu x)))
      (is (tracer? (act/relu x)))
      (is (tracer? (act/swiglu x x)))
      (is (tracer? (act/geglu x-m gw uw dw))))))
