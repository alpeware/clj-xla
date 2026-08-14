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
                                          [[:q [:tensor [1 8 16 256] :f32]]]
                                          (fn [q] (attn/apply-rope q [0])))]
                  (and (= "causal_attn" (:name g-attn))
                       (= "rope_test" (:name g-rope))
                       (seq (:eqns g-attn))
                       (seq (:eqns g-rope))))))

(defspec prop-update-kv-cache-invariants 50
  (prop/for-all [pos (gen/choose 0 10)
                 new-len (gen/choose 1 4)]
                (let [max-len 32
                      k-cache (t/->Tracer :k_cache [:tensor [1 4 max-len 64] :f32])
                      v-cache (t/->Tracer :v_cache [:tensor [1 4 max-len 64] :f32])
                      new-k (t/->Tracer :new_k [:tensor [1 4 new-len 64] :f32])
                      new-v (t/->Tracer :new_v [:tensor [1 4 new-len 64] :f32])
                      [[up-k up-v] [act-k act-v]] (attn/update-kv-cache [k-cache v-cache] new-k new-v pos)]
                  (and (= [:tensor [1 4 max-len 64] :f32] (:type up-k))
                       (= [:tensor [1 4 max-len 64] :f32] (:type up-v))
                       (= [:tensor [1 4 (+ pos new-len) 64] :f32] (:type act-k))
                       (= [:tensor [1 4 (+ pos new-len) 64] :f32] (:type act-v))))))

(deftest attention-tracer-test
  (testing "Causal self-attention returns Tracer on Tracer input"
    (let [x (t/->Tracer :x [:tensor [1 128 768] :f32])
          w (t/->Tracer :w [:tensor [768 2304] :f32])
          b (t/->Tracer :b [:tensor [2304] :f32])
          pw (t/->Tracer :pw [:tensor [768 768] :f32])
          pb (t/->Tracer :pb [:tensor [768] :f32])]
      (is (tracer? (attn/causal-self-attention x w b pw pb 12)))))

  (testing "apply-rope returns Tracer with matching 4D tensor shape"
    (let [q (t/->Tracer :q [:tensor [1 8 16 256] :f32])
          k (t/->Tracer :k [:tensor [1 4 16 256] :f32])
          q-rope (attn/apply-rope q [0])
          [q-r k-r] (attn/apply-rope q k [0])]
      (is (tracer? q-rope))
      (is (= [:tensor [1 8 16 256] :f32] (:type q-rope)))
      (is (tracer? q-r))
      (is (= [:tensor [1 8 16 256] :f32] (:type q-r)))
      (is (tracer? k-r))
      (is (= [:tensor [1 4 16 256] :f32] (:type k-r)))))

  (testing "causal-self-attention and gqa-causal-attention with past-kv return updated cache and active output"
    (let [x (t/->Tracer :x [:tensor [1 1 768] :f32])
          w (t/->Tracer :w [:tensor [768 2304] :f32])
          b (t/->Tracer :b [:tensor [2304] :f32])
          pw (t/->Tracer :pw [:tensor [768 768] :f32])
          pb (t/->Tracer :pb [:tensor [768] :f32])
          k-c (t/->Tracer :kc [:tensor [1 12 128 64] :f32])
          v-c (t/->Tracer :vc [:tensor [1 12 128 64] :f32])
          [out updated-kv] (attn/causal-self-attention x w b pw pb 12 [k-c v-c] 5)]
      (is (tracer? out))
      (is (= [:tensor [1 1 768] :f32] (:type out)))
      (is (vector? updated-kv)))))

(defspec prop-partial-rotary-rope 50
  (prop/for-all [rope-factor (gen/elements [0.25 0.5 1.0])]
                (let [q (t/->Tracer :q [:tensor [1 8 16 256] :f32])
                      q-rope (attn/apply-rope q nil [0] 256 1000000.0 nil nil rope-factor)]
                  (and (tracer? q-rope)
                       (= [:tensor [1 8 16 256] :f32] (:type q-rope))))))
