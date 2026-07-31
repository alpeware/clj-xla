(ns clj-xla.models.gpt2-test
  "Unit and generative tests for GPT-2 model architecture assembler."
  (:require [clj-xla.models.gpt2 :as gpt2]
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
