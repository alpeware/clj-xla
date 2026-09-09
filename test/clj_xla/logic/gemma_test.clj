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

(defspec prop-gemma4-kv-layer-ast-validity 20
  (prop/for-all [layer-idx (gen/choose 0 34)
                 max-seq-len (gen/elements [16 32 64])]
                (let [config (assoc (gemma/gemma4-config :e2b) :max-seq-len max-seq-len)
                      layer-ast (gemma/gemma4-kv-layer-ast layer-idx max-seq-len config)
                      expanded (expand/expand-ast {} layer-ast)]
                  (and (vector? layer-ast)
                       (seq expanded)
                       (every? ast/valid-node? expanded)))))

(deftest test-gemma4-kv-cache-step-lowering
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
        invars [[:x [:tensor [1 1] :i32]]
                [:pos [:tensor [1] :i32]]
                [:k_cache_in_0 [:tensor [1 max-seq-len 1 256] :f32]]
                [:v_cache_in_0 [:tensor [1 max-seq-len 1 256] :f32]]
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
        ast (gemma/gemma4-kv-model-ast config)
        graph (lower/ast->graph "gemma4_kv_step" invars ast [:logits :k_cache_out_0 :v_cache_out_0])]
    (is (shlo/validate-graph graph))
    (is (= [:logits :k_cache_out_0 :v_cache_out_0] (:outvars graph)))
    (let [ctx (xla/get-context)
          compiled (xla/compile-graph ctx graph)
          _ (is (some? compiled))
          x-data (int-array [42])
          pos-data (int-array [3])
          k-cache (float-array (repeat (* max-seq-len 256) 0.0))
          v-cache (float-array (repeat (* max-seq-len 256) 0.0))
          weights-data (mapv (fn [[_ [_ shape _]]]
                               (let [size (reduce * shape)]
                                 (float-array (repeat size 0.01))))
                             (subvec invars 4))
          all-inputs (into [x-data pos-data k-cache v-cache] weights-data)
          outs (apply xla/execute compiled all-inputs)
          [logits-buf k-out-buf v-out-buf] (if (vector? outs) outs [outs])
          logits (xla/to-host-slice logits-buf 0 vocab-size vocab-size)
          k-out (xla/to-host-slice k-out-buf 0 (* max-seq-len 256) (* max-seq-len 256))]
      (is (= vocab-size (count logits)))
      (is (some #(not= 0.0 %) (subvec (vec k-out) (* 3 256) (* 4 256))))
      (is (every? #(= 0.0 %) (subvec (vec k-out) 0 256)))
      (xla/destroy-buffer! logits-buf)
      (xla/destroy-buffer! k-out-buf)
      (xla/destroy-buffer! v-out-buf))))

(defspec prop-layer-is-global-schedules 50
  (prop/for-all [layer-idx (gen/choose 0 41)]
                (let [e2b-types nil
                      e4b-types (vec (mapcat (fn [_] (concat (repeat 5 "sliding_attention") ["full_attention"])) (range 7)))
                      e2b-res (gemma/layer-is-global? e2b-types layer-idx)
                      e4b-res (gemma/layer-is-global? e4b-types layer-idx)]
                  (and (= e2b-res (zero? (mod (inc layer-idx) 5)))
                       (= e4b-res (= 5 (mod layer-idx 6)))))))

(deftest test-gemma4-prefill-lowering
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
        graph (lower/ast->graph "gemma4_prefill" invars ast [:logits :k_ro_0 :v_heads_0])]
    (is (shlo/validate-graph graph))
    (is (= [:logits :k_ro_0 :v_heads_0] (:outvars graph)))
    (let [ctx (xla/get-context)
          compiled (xla/compile-graph ctx graph)]
      (is (some? compiled)))))

