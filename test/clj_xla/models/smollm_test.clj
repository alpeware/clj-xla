(ns clj-xla.models.smollm-test
  "Unit and generative tests for SmolLM model configuration and tracing."
  (:require [clj-xla.models.smollm :refer [smollm-block smollm-config weight-key-map]]
            [clj-xla.tensor :as t :refer [tracer?]]
            [clojure.test :refer [deftest is testing]]
            [clojure.test.check.clojure-test :refer [defspec]]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]))

(defspec prop-smollm-config-defaults 50
  (prop/for-all [vocab-sz (gen/choose 1000 60000)]
                (let [cfg (smollm-config {:vocab-size vocab-sz})]
                  (and (= vocab-sz (:vocab-size cfg))
                       (= 576 (:n-embd cfg))
                       (= 30 (:n-layer cfg))))))

(deftest smollm-weight-key-map-test
  (testing "Weight key map generates exact HuggingFace SmolLM key names"
    (let [km (weight-key-map 0)]
      (is (= "model.layers.0.input_layernorm.weight" (:input-ln-w km)))
      (is (= "model.layers.0.self_attn.q_proj.weight" (:q-w km)))
      (is (= "model.layers.0.mlp.gate_proj.weight" (:gate-w km))))))

(deftest smollm-block-tracer-test
  (testing "SmolLM block returns Tracer on Tracer input"
    (let [x (t/->Tracer :x [:tensor [1 16 576] :f32])
          weights {:input-ln-w (t/->Tracer :in_ln [:tensor [576] :f32])
                   :q-w (t/->Tracer :qw [:tensor [576 576] :f32])
                   :k-w (t/->Tracer :kw [:tensor [576 192] :f32])
                   :v-w (t/->Tracer :vw [:tensor [576 192] :f32])
                   :o-w (t/->Tracer :ow [:tensor [576 576] :f32])
                   :post-attn-ln-w (t/->Tracer :post_ln [:tensor [576] :f32])
                   :gate-w (t/->Tracer :gw [:tensor [576 1536] :f32])
                   :up-w (t/->Tracer :uw [:tensor [576 1536] :f32])
                   :down-w (t/->Tracer :dw [:tensor [1536 576] :f32])}]
      (is (tracer? (smollm-block x weights 9 3 [0]))))))

(defspec prop-smollm-kv-cache-forward-invariants 30
  (prop/for-all [pos (gen/choose 0 5)]
                (let [x (t/->Tracer :x [:tensor [1 1] :i32])
                      pos-ids [0]
                      embed (t/->Tracer :emb [:tensor [49152 576] :f32])
                      final-norm (t/->Tracer :fn [:tensor [576] :f32])
                      lm-head (t/->Tracer :lm [:tensor [49152 576] :f32])
                      layer-w {:input-ln-w (t/->Tracer :in_ln [:tensor [576] :f32])
                               :q-w (t/->Tracer :qw [:tensor [576 576] :f32])
                               :k-w (t/->Tracer :kw [:tensor [576 192] :f32])
                               :v-w (t/->Tracer :vw [:tensor [576 192] :f32])
                               :o-w (t/->Tracer :ow [:tensor [576 576] :f32])
                               :post-attn-ln-w (t/->Tracer :post_ln [:tensor [576] :f32])
                               :gate-w (t/->Tracer :gw [:tensor [576 1536] :f32])
                               :up-w (t/->Tracer :uw [:tensor [576 1536] :f32])
                               :down-w (t/->Tracer :dw [:tensor [1536 576] :f32])}
                      kv-caches [[(t/->Tracer :kc [:tensor [1 3 32 64] :f32])
                                  (t/->Tracer :vc [:tensor [1 3 32 64] :f32])]]
                      [logits updated-kv] (clj-xla.models.smollm/full-smollm-forward x embed [layer-w] final-norm lm-head pos-ids kv-caches pos)]
                  (and (tracer? logits)
                       (= 1 (count updated-kv))))))

