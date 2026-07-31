(ns clj-xla.nn.attention-test
  "Unit and generative tests for self-attention mechanisms and RoPE positioning."
  (:require [clj-xla.nn.attention :as attn]
            [clj-xla.tensor :as t :refer [tracer?]]
            [clj-xla.trace :refer [trace-graph]]
            [clojure.test :refer [deftest is testing]]
            [clojure.test.check.clojure-test :refer [defspec]]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]))

(defspec prop-attention-tracing 50
  (prop/for-all [_seq-len (gen/choose 1 32)]
                (let [g-attn (trace-graph "causal_attn"
                                          [[:x [:tensor [1 128 768] :f32]]
                                           [:c_attn_w [:tensor [768 2304] :f32]]
                                           [:c_attn_b [:tensor [2304] :f32]]
                                           [:c_proj_w [:tensor [768 768] :f32]]
                                           [:c_proj_b [:tensor [768] :f32]]]
                                          (fn [x w b pw pb] (attn/causal-self-attention x w b pw pb 12)))
                      g-rope (trace-graph "rope_test"
                                          [[:q [:tensor [1 16 96] :f32]]]
                                          (fn [q] (attn/apply-rope q [0])))]
                  (and (= "causal_attn" (:name g-attn))
                       (= "rope_test" (:name g-rope))
                       (seq (:eqns g-attn))
                       (seq (:eqns g-rope))))))

(deftest attention-tracer-test
  (testing "Causal self-attention returns Tracer on Tracer input"
    (let [x (t/->Tracer :x [:tensor [1 128 768] :f32])
          w (t/->Tracer :w [:tensor [768 2304] :f32])
          b (t/->Tracer :b [:tensor [2304] :f32])
          pw (t/->Tracer :pw [:tensor [768 768] :f32])
          pb (t/->Tracer :pb [:tensor [768] :f32])]
      (is (tracer? (attn/causal-self-attention x w b pw pb 12)))
      (is (tracer? (attn/apply-rope x [0]))))))
