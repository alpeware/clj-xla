(ns clj-xla.logic.gemma-test
  "Unit and generative tests for declarative Tensor Logic Gemma 4 model architecture."
  (:require [clj-xla.core :as xla]
            [clj-xla.logic.ast :as ast]
            [clj-xla.logic.expand :as expand]
            [clj-xla.logic.lower :as lower]
            [clj-xla.logic.models.gemma :as gemma]
            [clj-xla.stablehlo :as shlo]
            [clojure.test :refer [deftest is]]
            [clojure.test.check.clojure-test :refer [defspec]]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]))

(defspec prop-gemma4-layer-ast-validity 20
  (prop/for-all [layer-idx (gen/choose 0 34)
                 seq-len (gen/elements [8 16 32])]
                (let [config (gemma/gemma4-config :e2b)
                      layer-ast (gemma/gemma4-layer-ast layer-idx seq-len config)
                      expanded (expand/expand-ast {} layer-ast)]
                  (and (vector? layer-ast)
                       (seq expanded)
                       (every? ast/valid-node? expanded)))))

(deftest test-gemma4-single-layer-lowering
  (let [num-layers 1
        max-seq-len 8
        vocab-size 1000
        hidden-dim 1536
        intermediate-dim 6144
        pl-dim 256
        total-pl-dim 256
        config {:vocab-size vocab-size
                :hidden-dim hidden-dim
                :intermediate-dim intermediate-dim
                :pl-dim pl-dim
                :total-pl-dim total-pl-dim
                :num-layers num-layers
                :num-heads 8
                :num-kv-heads 1
                :head-dim 256
                :max-seq-len max-seq-len}
        invars [[:x [:tensor [1 max-seq-len] :i32]]
                [:embed_tokens [:tensor [vocab-size hidden-dim] :f32]]
                [:embed_tokens_per_layer [:tensor [vocab-size total-pl-dim] :f32]]
                [:per_layer_model_projection [:tensor [total-pl-dim hidden-dim] :f32]]
                [:per_layer_projection_norm [:tensor [pl-dim] :f32]]
                [:final_norm_w [:tensor [hidden-dim] :f32]]
                [:input_ln_w_0 [:tensor [hidden-dim] :f32]]
                [:layer_scalar_0 [:tensor [1] :f32]]
                [:q_w_0 [:tensor [2048 hidden-dim] :f32]]
                [:k_w_0 [:tensor [256 hidden-dim] :f32]]
                [:v_w_0 [:tensor [256 hidden-dim] :f32]]
                [:o_w_0 [:tensor [hidden-dim 2048] :f32]]
                [:q_norm_w_0 [:tensor [256] :f32]]
                [:k_norm_w_0 [:tensor [256] :f32]]
                [:post_attn_ln_w_0 [:tensor [hidden-dim] :f32]]
                [:pre_mlp_ln_w_0 [:tensor [hidden-dim] :f32]]
                [:post_mlp_ln_w_0 [:tensor [hidden-dim] :f32]]
                [:gate_w_0 [:tensor [intermediate-dim hidden-dim] :f32]]
                [:up_w_0 [:tensor [intermediate-dim hidden-dim] :f32]]
                [:down_w_0 [:tensor [hidden-dim intermediate-dim] :f32]]
                [:per_layer_gate_w_0 [:tensor [pl-dim hidden-dim] :f32]]
                [:per_layer_proj_w_0 [:tensor [hidden-dim pl-dim] :f32]]
                [:post_per_layer_norm_w_0 [:tensor [hidden-dim] :f32]]]
        ast (gemma/gemma4-model-ast config)
        graph (lower/ast->graph "gemma4_single_layer" invars ast #{:logits})]
    (is (shlo/validate-graph graph))
    (is (= [:logits] (:outvars graph)))
    (let [ctx (xla/get-context)
          compiled (xla/compile-graph ctx graph)]
      (is (some? compiled)))))

(deftest test-gemma4-last-token-only-lowering
  (let [num-layers 1
        max-seq-len 8
        vocab-size 1000
        hidden-dim 1536
        intermediate-dim 6144
        pl-dim 256
        total-pl-dim 256
        config {:vocab-size vocab-size
                :hidden-dim hidden-dim
                :intermediate-dim intermediate-dim
                :pl-dim pl-dim
                :total-pl-dim total-pl-dim
                :num-layers num-layers
                :num-heads 8
                :num-kv-heads 1
                :head-dim 256
                :max-seq-len max-seq-len
                :last-token-only? true}
        invars [[:x [:tensor [1 max-seq-len] :i32]]
                [:pos [:tensor [1] :i32]]
                [:embed_tokens [:tensor [vocab-size hidden-dim] :f32]]
                [:embed_tokens_per_layer [:tensor [vocab-size total-pl-dim] :f32]]
                [:per_layer_model_projection [:tensor [total-pl-dim hidden-dim] :f32]]
                [:per_layer_projection_norm [:tensor [pl-dim] :f32]]
                [:final_norm_w [:tensor [hidden-dim] :f32]]
                [:input_ln_w_0 [:tensor [hidden-dim] :f32]]
                [:layer_scalar_0 [:tensor [1] :f32]]
                [:q_w_0 [:tensor [2048 hidden-dim] :f32]]
                [:k_w_0 [:tensor [256 hidden-dim] :f32]]
                [:v_w_0 [:tensor [256 hidden-dim] :f32]]
                [:o_w_0 [:tensor [hidden-dim 2048] :f32]]
                [:q_norm_w_0 [:tensor [256] :f32]]
                [:k_norm_w_0 [:tensor [256] :f32]]
                [:post_attn_ln_w_0 [:tensor [hidden-dim] :f32]]
                [:pre_mlp_ln_w_0 [:tensor [hidden-dim] :f32]]
                [:post_mlp_ln_w_0 [:tensor [hidden-dim] :f32]]
                [:gate_w_0 [:tensor [intermediate-dim hidden-dim] :f32]]
                [:up_w_0 [:tensor [intermediate-dim hidden-dim] :f32]]
                [:down_w_0 [:tensor [hidden-dim intermediate-dim] :f32]]
                [:per_layer_gate_w_0 [:tensor [pl-dim hidden-dim] :f32]]
                [:per_layer_proj_w_0 [:tensor [hidden-dim pl-dim] :f32]]
                [:post_per_layer_norm_w_0 [:tensor [hidden-dim] :f32]]]
        ast (gemma/gemma4-model-ast config)
        graph (lower/ast->graph "gemma4_last_token" invars ast #{:logits})]
    (is (shlo/validate-graph graph))
    (is (= [:logits] (:outvars graph)))
    (let [ctx (xla/get-context)
          compiled (xla/compile-graph ctx graph)]
      (is (some? compiled)))))

(defspec prop-gemma4-config-invariants 20
  (prop/for-all [model-kw (gen/elements [:e2b :e4b :12b :26b :31b])]
                (let [cfg (gemma/gemma4-config model-kw)]
                  (and (pos? (:hidden-dim cfg))
                       (pos? (:num-layers cfg))
                       (pos? (:num-heads cfg))
                       (pos? (:head-dim cfg))
                       (pos? (:vocab-size cfg))))))

(defspec prop-gemma4-weight-key-map 20
  (prop/for-all [layer-idx (gen/choose 0 34)]
                (let [kmap (gemma/gemma4-weight-key-map layer-idx "layers.")]
                  (and (string? (:q-w kmap))
                       (string? (:k-w kmap))
                       (string? (:v-w kmap))
                       (string? (:o-w kmap))
                       (string? (:gate-w kmap))
                       (string? (:up-w kmap))
                       (string? (:down-w kmap))))))
