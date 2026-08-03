(ns clj-xla.models.gemma-test
  "Unit and generative tests for Gemma 2B model configuration, key mapping, and tracing."
  (:require [clj-xla.models.gemma :refer [full-gemma-forward gemma-block gemma-config weight-key-map]]
            [clj-xla.tensor :as t :refer [tracer?]]
            [clj-xla.trace :refer [trace-graph]]
            [clojure.test :refer [deftest is testing]]
            [clojure.test.check.clojure-test :refer [defspec]]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]))

(defspec prop-gemma-config-defaults 50
  (prop/for-all [vocab-sz (gen/choose 1000 300000)]
                (let [cfg (gemma-config {:vocab-size vocab-sz})]
                  (and (= vocab-sz (:vocab-size cfg))
                       (= 2048 (:hidden-size cfg))
                       (= 18 (:num-hidden-layers cfg))
                       (= 8 (:num-attention-heads cfg))
                       (= 1 (:num-key-value-heads cfg))
                       (= 256 (:head-dim cfg))))))

(deftest gemma-weight-key-map-test
  (testing "Weight key map generates exact HuggingFace Gemma key names including Gemma 2 RMSNorms"
    (let [km (weight-key-map 0)]
      (is (= "model.layers.0.input_layernorm.weight" (:input-ln-w km)))
      (is (= "model.layers.0.self_attn.q_proj.weight" (:q-w km)))
      (is (= "model.layers.0.self_attn.k_proj.weight" (:k-w km)))
      (is (= "model.layers.0.self_attn.v_proj.weight" (:v-w km)))
      (is (= "model.layers.0.self_attn.o_proj.weight" (:o-w km)))
      (is (= "model.layers.0.post_attention_layernorm.weight" (:post-attn-ln-w km)))
      (is (= "model.layers.0.pre_feedforward_layernorm.weight" (:pre-mlp-ln-w km)))
      (is (= "model.layers.0.post_feedforward_layernorm.weight" (:post-mlp-ln-w km)))
      (is (= "model.layers.0.mlp.gate_proj.weight" (:gate-w km)))
      (is (= "model.layers.0.mlp.up_proj.weight" (:up-w km)))
      (is (= "model.layers.0.mlp.down_proj.weight" (:down-w km))))))

(deftest gemma-block-tracer-test
  (testing "Gemma 2 block returns Tracer on Tracer input"
    (let [x (t/->Tracer :x [:tensor [1 16 2304] :f32])
          weights {:input-ln-w (t/->Tracer :in_ln [:tensor [2304] :f32])
                   :q-w (t/->Tracer :qw [:tensor [2048 2304] :f32])
                   :k-w (t/->Tracer :kw [:tensor [1024 2304] :f32])
                   :v-w (t/->Tracer :vw [:tensor [1024 2304] :f32])
                   :o-w (t/->Tracer :ow [:tensor [2304 2048] :f32])
                   :post-attn-ln-w (t/->Tracer :post_attn_ln [:tensor [2304] :f32])
                   :pre-mlp-ln-w (t/->Tracer :pre_mlp_ln [:tensor [2304] :f32])
                   :post-mlp-ln-w (t/->Tracer :post_mlp_ln [:tensor [2304] :f32])
                   :gate-w (t/->Tracer :gw [:tensor [9216 2304] :f32])
                   :up-w (t/->Tracer :uw [:tensor [9216 2304] :f32])
                   :down-w (t/->Tracer :dw [:tensor [2304 9216] :f32])}]
      (is (tracer? (gemma-block x weights 8 4 [0]))))))

(deftest full-gemma-forward-tracer-test
  (testing "Full Gemma 2 forward pass traces valid computation graph"
    (let [layer-w {:input-ln-w (t/->Tracer :in_ln [:tensor [2304] :f32])
                   :q-w (t/->Tracer :qw [:tensor [2048 2304] :f32])
                   :k-w (t/->Tracer :kw [:tensor [1024 2304] :f32])
                   :v-w (t/->Tracer :vw [:tensor [1024 2304] :f32])
                   :o-w (t/->Tracer :ow [:tensor [2304 2048] :f32])
                   :post-attn-ln-w (t/->Tracer :post_attn_ln [:tensor [2304] :f32])
                   :pre-mlp-ln-w (t/->Tracer :pre_mlp_ln [:tensor [2304] :f32])
                   :post-mlp-ln-w (t/->Tracer :post_mlp_ln [:tensor [2304] :f32])
                   :gate-w (t/->Tracer :gw [:tensor [9216 2304] :f32])
                   :up-w (t/->Tracer :uw [:tensor [9216 2304] :f32])
                   :down-w (t/->Tracer :dw [:tensor [2304 9216] :f32])}
          graph (trace-graph "gemma2_test"
                             [[:x [:tensor [1 16] :i32]]
                              [:emb [:tensor [256000 2304] :f32]]
                              [:fn_norm [:tensor [2304] :f32]]]
                             (fn [x emb fn-norm]
                               (full-gemma-forward x emb [layer-w] fn-norm [0])))]
      (is (= "gemma2_test" (:name graph)))
      (is (seq (:eqns graph)))))

  (testing "Full Gemma forward pass with kv-caches returns [logits updated-kv-caches]"
    (let [layer-w {:input-ln-w (t/->Tracer :in_ln [:tensor [2304] :f32])
                   :q-w (t/->Tracer :qw [:tensor [2048 2304] :f32])
                   :k-w (t/->Tracer :kw [:tensor [1024 2304] :f32])
                   :v-w (t/->Tracer :vw [:tensor [1024 2304] :f32])
                   :o-w (t/->Tracer :ow [:tensor [2304 2048] :f32])
                   :post-attn-ln-w (t/->Tracer :post_attn_ln [:tensor [2304] :f32])
                   :pre-mlp-ln-w (t/->Tracer :pre_mlp_ln [:tensor [2304] :f32])
                   :post-mlp-ln-w (t/->Tracer :post_mlp_ln [:tensor [2304] :f32])
                   :gate-w (t/->Tracer :gw [:tensor [9216 2304] :f32])
                   :up-w (t/->Tracer :uw [:tensor [9216 2304] :f32])
                   :down-w (t/->Tracer :dw [:tensor [2304 9216] :f32])}
          kv-caches [[(t/->Tracer :kc [:tensor [1 4 32 256] :f32])
                      (t/->Tracer :vc [:tensor [1 4 32 256] :f32])]]
          x (t/->Tracer :x [:tensor [1 1] :i32])
          emb (t/->Tracer :emb [:tensor [256000 2304] :f32])
          fn-norm (t/->Tracer :fn_norm [:tensor [2304] :f32])
          [logits updated-kv] (full-gemma-forward x emb [layer-w] fn-norm [0] 8 4 kv-caches 5)]
      (is (tracer? logits))
      (is (= 1 (count updated-kv))))))

