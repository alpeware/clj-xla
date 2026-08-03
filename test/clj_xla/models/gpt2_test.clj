(ns clj-xla.models.gpt2-test
  "Unit and generative tests for GPT-2 model architecture assembler."
  (:require [clj-xla.models.gpt2 :as gpt2]
            [clj-xla.tensor :as t]
            [clj-xla.trace :refer [trace-graph]]
            [clojure.test :refer [deftest is testing]]
            [clojure.test.check.clojure-test :refer [defspec]]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]))

(defspec prop-gpt2-config-defaults 50
  (prop/for-all [n-layer (gen/choose 1 12)
                 n-head (gen/choose 1 12)]
                (let [cfg (gpt2/gpt2-config {:n-layer n-layer :n-head n-head})]
                  (and (= n-layer (:n-layer cfg))
                       (= n-head (:n-head cfg))
                       (= 768 (:n-embd cfg))
                       (= 50257 (:vocab-size cfg))))))

(deftest gpt2-block-trace-test
  (testing "Symbolic tracing of GPT-2 Transformer block"
    (let [graph (trace-graph "gpt2_block_trace"
                             [[:x [:tensor [1 128 768] :f32]]
                              [:ln1_g [:tensor [768] :f32]]
                              [:ln1_b [:tensor [768] :f32]]
                              [:c_attn_w [:tensor [768 2304] :f32]]
                              [:c_attn_b [:tensor [2304] :f32]]
                              [:c_proj_w [:tensor [768 768] :f32]]
                              [:c_proj_b [:tensor [768] :f32]]
                              [:ln2_g [:tensor [768] :f32]]
                              [:ln2_b [:tensor [768] :f32]]
                              [:mlp_fc_w [:tensor [768 3072] :f32]]
                              [:mlp_fc_b [:tensor [3072] :f32]]
                              [:mlp_proj_w [:tensor [3072 768] :f32]]
                              [:mlp_proj_b [:tensor [768] :f32]]]
                             (fn [x ln1g ln1b cw cb pw pb ln2g ln2b fcw fcb pw2 pb2]
                               (gpt2/gpt2-block x {:ln1-g ln1g :ln1-b ln1b
                                                   :c-attn-w cw :c-attn-b cb
                                                   :c-proj-w pw :c-proj-b pb
                                                   :ln2-g ln2g :ln2-b ln2b
                                                   :mlp-fc-w fcw :mlp-fc-b fcb
                                                   :mlp-proj-w pw2 :mlp-proj-b pb2} 12)))]
      (is (= "gpt2_block_trace" (:name graph)))
      (is (pos? (count (:eqns graph)))))))

(defspec prop-gpt2-kv-cache-forward-invariants 30
  (prop/for-all [pos (gen/choose 0 5)]
                (let [x (t/->Tracer :x [:tensor [1 1] :i32])
                      pos-ids (t/->Tracer :pos [:tensor [1 1] :i32])
                      ln-f-g (t/->Tracer :g [:tensor [768] :f32])
                      ln-f-b (t/->Tracer :b [:tensor [768] :f32])
                      wte (t/->Tracer :wte [:tensor [50257 768] :f32])
                      wpe (t/->Tracer :wpe [:tensor [1024 768] :f32])
                      layer-w {:ln1-g (t/->Tracer :g1 [:tensor [768] :f32])
                               :ln1-b (t/->Tracer :b1 [:tensor [768] :f32])
                               :c-attn-w (t/->Tracer :cw [:tensor [768 2304] :f32])
                               :c-attn-b (t/->Tracer :cb [:tensor [2304] :f32])
                               :c-proj-w (t/->Tracer :pw [:tensor [768 768] :f32])
                               :c-proj-b (t/->Tracer :pb [:tensor [768] :f32])
                               :ln2-g (t/->Tracer :g2 [:tensor [768] :f32])
                               :ln2-b (t/->Tracer :b2 [:tensor [768] :f32])
                               :mlp-fc-w (t/->Tracer :fw [:tensor [768 3072] :f32])
                               :mlp-fc-b (t/->Tracer :fb [:tensor [3072] :f32])
                               :mlp-proj-w (t/->Tracer :pw2 [:tensor [3072 768] :f32])
                               :mlp-proj-b (t/->Tracer :pb2 [:tensor [768] :f32])}
                      kv-caches [[(t/->Tracer :kc [:tensor [1 12 32 64] :f32])
                                  (t/->Tracer :vc [:tensor [1 12 32 64] :f32])]]
                      [logits updated-kv] (gpt2/full-gpt2-forward x pos-ids ln-f-g ln-f-b wte wpe [layer-w] kv-caches pos)]
                  (and (t/tracer? logits)
                       (= 1 (count updated-kv))))))


