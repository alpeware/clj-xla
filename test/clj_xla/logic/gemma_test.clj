(ns clj-xla.logic.gemma-test
  "Unit and generative tests for declarative Tensor Logic Gemma 4 model architecture."
  (:require [clj-xla.core :as xla]
            [clj-xla.pjrt :as pjrt]
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

(defspec prop-gemma4-int8-layer-ast-validity 20
  (prop/for-all [layer-idx (gen/choose 0 34)
                 seq-len (gen/elements [8 16 32])]
                (let [config (assoc (gemma/gemma4-config :e2b) :is-int8 true)
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

(deftest test-gemma4-single-layer-int8-lowering
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
                :is-int8 true
                :norm-dtype :f32}
        invars [[:x [:tensor [1 max-seq-len] :i32]]
                [:embed_tokens [:tensor [vocab-size hidden-dim] :f32]]
                [:embed_tokens_per_layer [:tensor [vocab-size total-pl-dim] :f32]]
                [:per_layer_model_projection [:tensor [total-pl-dim hidden-dim] :f32]]
                [:per_layer_projection_norm [:tensor [pl-dim] :f32]]
                [:final_norm_w [:tensor [hidden-dim] :f32]]
                [:input_ln_w_0 [:tensor [hidden-dim] :f32]]
                [:layer_scalar_0 [:tensor [1] :f32]]
                [:q_w_0 [:tensor [2048 hidden-dim] :i8]]
                [:q_scale_0 [:tensor [2048] :f32]]
                [:k_w_0 [:tensor [256 hidden-dim] :i8]]
                [:k_scale_0 [:tensor [256] :f32]]
                [:v_w_0 [:tensor [256 hidden-dim] :i8]]
                [:v_scale_0 [:tensor [256] :f32]]
                [:o_w_0 [:tensor [hidden-dim 2048] :i8]]
                [:o_scale_0 [:tensor [hidden-dim] :f32]]
                [:q_norm_w_0 [:tensor [256] :f32]]
                [:k_norm_w_0 [:tensor [256] :f32]]
                [:post_attn_ln_w_0 [:tensor [hidden-dim] :f32]]
                [:pre_mlp_ln_w_0 [:tensor [hidden-dim] :f32]]
                [:post_mlp_ln_w_0 [:tensor [hidden-dim] :f32]]
                [:gate_w_0 [:tensor [intermediate-dim hidden-dim] :i8]]
                [:gate_scale_0 [:tensor [intermediate-dim] :f32]]
                [:up_w_0 [:tensor [intermediate-dim hidden-dim] :i8]]
                [:up_scale_0 [:tensor [intermediate-dim] :f32]]
                [:down_w_0 [:tensor [hidden-dim intermediate-dim] :i8]]
                [:down_scale_0 [:tensor [hidden-dim] :f32]]
                [:per_layer_gate_w_0 [:tensor [pl-dim hidden-dim] :f32]]
                [:per_layer_proj_w_0 [:tensor [hidden-dim pl-dim] :f32]]
                [:post_per_layer_norm_w_0 [:tensor [hidden-dim] :f32]]]
        ast (gemma/gemma4-model-ast config)
        graph (lower/ast->graph "gemma4_single_layer_int8" invars ast #{:logits})]
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

(defspec prop-gemma4-kv-int8-layer-ast-validity 20
  (prop/for-all [layer-idx (gen/choose 0 34)
                 max-seq-len (gen/elements [16 32 64])]
                (let [config (assoc (gemma/gemma4-config :e2b) :max-seq-len max-seq-len :is-int8 true)
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

(deftest test-gemma4-kv-cache-step-ring-buffer-lowering
  (let [num-layers 1
        max-seq-len 16
        window 4
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
                :sliding-window window}
        invars [[:x [:tensor [1 1] :i32]]
                [:pos [:tensor [1] :i32]]]
        invars (into invars
                     [[:k_cache_in_0 [:tensor [1 window 1 256] :f32]]
                      [:v_cache_in_0 [:tensor [1 window 1 256] :f32]]
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
                      [:post_per_layer_norm_w_0 [:tensor [hidden-dim] :f32]]])
        ast (gemma/gemma4-kv-model-ast config)
        graph (lower/ast->graph "gemma4_kv_ring_step" invars ast [:logits :k_cache_out_0 :v_cache_out_0])]
    (is (shlo/validate-graph graph))
    (let [ctx (xla/get-context)
          compiled (xla/compile-graph ctx graph)
          x-data (int-array [42])
          pos-data (int-array [6]) ;; 6 mod 4 = 2
          k-cache (float-array (repeat (* window 256) 0.0))
          v-cache (float-array (repeat (* window 256) 0.0))
          weights-data (mapv (fn [[_ [_ shape _]]]
                               (let [size (reduce * shape)]
                                 (float-array (repeat size 0.01))))
                             (subvec invars 4))
          all-inputs (into [x-data pos-data k-cache v-cache] weights-data)
          outs (apply xla/execute compiled all-inputs)
          [logits-buf k-out-buf v-out-buf] (if (vector? outs) outs [outs])
          logits (xla/to-host-slice logits-buf 0 vocab-size vocab-size)
          k-out (xla/to-host-slice k-out-buf 0 (* window 256) (* window 256))]
      (is (= vocab-size (count logits)))
      ;; Slot 2 (indices 512 to 767) should be populated because 6 mod 4 = 2!
      (is (some #(not= 0.0 %) (subvec (vec k-out) (* 2 256) (* 3 256))))
      ;; Slot 0 (indices 0 to 255) and slot 1 (indices 256 to 511) should remain 0.0
      (is (every? #(= 0.0 %) (subvec (vec k-out) 0 256)))
      (is (every? #(= 0.0 %) (subvec (vec k-out) 256 512)))
      (xla/destroy-buffer! logits-buf)
      (xla/destroy-buffer! k-out-buf)
      (xla/destroy-buffer! v-out-buf))))

(deftest test-gemma4-kv-cache-step-int8-lowering
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
                :is-int8 true
                :norm-dtype :f32}
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
                [:q_w_0 [:tensor [2048 hidden-dim] :i8]]
                [:q_scale_0 [:tensor [2048] :f32]]
                [:k_w_0 [:tensor [256 hidden-dim] :i8]]
                [:k_scale_0 [:tensor [256] :f32]]
                [:v_w_0 [:tensor [256 hidden-dim] :i8]]
                [:v_scale_0 [:tensor [256] :f32]]
                [:o_w_0 [:tensor [hidden-dim 2048] :i8]]
                [:o_scale_0 [:tensor [hidden-dim] :f32]]
                [:q_norm_w_0 [:tensor [256] :f32]]
                [:k_norm_w_0 [:tensor [256] :f32]]
                [:post_attn_ln_w_0 [:tensor [hidden-dim] :f32]]
                [:pre_mlp_ln_w_0 [:tensor [hidden-dim] :f32]]
                [:post_mlp_ln_w_0 [:tensor [hidden-dim] :f32]]
                [:gate_w_0 [:tensor [intermediate-dim hidden-dim] :i8]]
                [:gate_scale_0 [:tensor [intermediate-dim] :f32]]
                [:up_w_0 [:tensor [intermediate-dim hidden-dim] :i8]]
                [:up_scale_0 [:tensor [intermediate-dim] :f32]]
                [:down_w_0 [:tensor [hidden-dim intermediate-dim] :i8]]
                [:down_scale_0 [:tensor [hidden-dim] :f32]]
                [:per_layer_gate_w_0 [:tensor [pl-dim hidden-dim] :f32]]
                [:per_layer_proj_w_0 [:tensor [hidden-dim pl-dim] :f32]]
                [:post_per_layer_norm_w_0 [:tensor [hidden-dim] :f32]]]
        ast (gemma/gemma4-kv-model-ast config)
        graph (lower/ast->graph "gemma4_kv_step_int8" invars ast [:logits :k_cache_out_0 :v_cache_out_0])]
    (is (shlo/validate-graph graph))
    (is (= [:logits :k_cache_out_0 :v_cache_out_0] (:outvars graph)))
    (let [ctx (xla/get-context)
          compiled (xla/compile-graph ctx graph)
          _ (is (some? compiled))
          x-data (int-array [42])
          pos-data (int-array [3])
          k-cache (float-array (repeat (* max-seq-len 256) 0.0))
          v-cache (float-array (repeat (* max-seq-len 256) 0.0))
          weights-data (mapv (fn [[_ [_ shape dtype]]]
                               (let [size (reduce * shape)]
                                 (if (= dtype :i8)
                                   (byte-array (repeat size (byte 1)))
                                   (float-array (repeat size 0.01)))))
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

(defspec prop-chunked-attention-ast-validity 20
  (prop/for-all [max-seq-len (gen/elements [64 128 256])
                 chunk-size (gen/elements [32 64])
                 num-heads (gen/elements [4 8 16])
                 num-kv-heads (gen/elements [1 2 4])
                 head-dim (gen/elements [64 128 256])]
                (let [group-size (quot num-heads num-kv-heads)
                      num-heads (* num-kv-heads group-size)
                      node [:chunked-attention [:ctx :b :p :h :dh]
                            [:q_ro :b :p :h :dh]
                            [:k_cache :b :kv-s :kvh :dh]
                            [:v_cache :b :kv-s :kvh :dh]
                            {:pos :pos
                             :chunk-size chunk-size
                             :head-dim head-dim
                             :num-heads num-heads
                             :num-kv-heads num-kv-heads
                             :max-seq-len max-seq-len}]]
                  (ast/valid-node? node))))

(defn- test-standard-attention-graph
  ([max-seq-len num-heads num-kv-heads head-dim]
   (test-standard-attention-graph max-seq-len num-heads num-kv-heads head-dim nil))
  ([max-seq-len num-heads num-kv-heads head-dim window]
   (let [group-size (quot num-heads num-kv-heads)
         invars [[:pos [:tensor [1] :i32]]
                 [:q_ro [:tensor [1 1 num-heads head-dim] :f32]]
                 [:k_cache [:tensor [1 max-seq-len num-kv-heads head-dim] :f32]]
                 [:v_cache [:tensor [1 max-seq-len num-kv-heads head-dim] :f32]]]
         outvars [:ctx]
         mask-eqns (if window
                     [{:op :stablehlo/iota :outvars [:iota_1d] :attrs {:len max-seq-len :dtype :i32 :iota_dimension 0}}
                      {:op :stablehlo/broadcast_in_dim :invars [:iota_1d] :outvars [:iota_4d]
                       :attrs {:broadcast_dimensions [3] :target_shape [1 1 1 max-seq-len]}}
                      {:op :stablehlo/reshape :invars [:pos] :outvars [:pos_s] :attrs {:shape []}}
                      {:op :stablehlo/broadcast_in_dim :invars [:pos_s] :outvars [:pos_4d]
                       :attrs {:broadcast_dimensions [] :target_shape [1 1 1 max-seq-len]}}
                      {:op :stablehlo/compare :invars [:iota_4d :pos_4d] :outvars [:cmp_fut] :attrs {:comparison_direction "GT"}}
                      {:op :stablehlo/constant :value (int window) :type [:tensor [] :i32] :outvars [:c_win]}
                      {:op :stablehlo/subtract :invars [:pos_s :c_win] :outvars [:p_sub]}
                      {:op :stablehlo/constant :value 1 :type [:tensor [] :i32] :outvars [:c_one]}
                      {:op :stablehlo/add :invars [:p_sub :c_one] :outvars [:min_p]}
                      {:op :stablehlo/broadcast_in_dim :invars [:min_p] :outvars [:min_p_4d]
                       :attrs {:broadcast_dimensions [] :target_shape [1 1 1 max-seq-len]}}
                      {:op :stablehlo/compare :invars [:iota_4d :min_p_4d] :outvars [:cmp_old] :attrs {:comparison_direction "LT"}}
                      {:op :stablehlo/or :invars [:cmp_fut :cmp_old] :outvars [:cmp_mask]}
                      {:op :stablehlo/constant :value -10000.0 :type [:tensor [] :f32] :outvars [:c_neg]}
                      {:op :stablehlo/broadcast_in_dim :invars [:c_neg] :outvars [:c_neg_4d]
                       :attrs {:broadcast_dimensions [] :target_shape [1 1 1 max-seq-len]}}
                      {:op :stablehlo/constant :value 0.0 :type [:tensor [] :f32] :outvars [:c_zero]}
                      {:op :stablehlo/broadcast_in_dim :invars [:c_zero] :outvars [:c_zero_4d]
                       :attrs {:broadcast_dimensions [] :target_shape [1 1 1 max-seq-len]}}
                      {:op :stablehlo/select :invars [:cmp_mask :c_neg_4d :c_zero_4d] :outvars [:mask_4d]}
                      {:op :stablehlo/broadcast_in_dim :invars [:mask_4d] :outvars [:mask_bcast]
                       :attrs {:broadcast_dimensions [0 1 2 3] :target_shape [1 num-heads 1 max-seq-len]}}]
                     [{:op :stablehlo/iota :outvars [:iota_1d] :attrs {:len max-seq-len :dtype :i32 :iota_dimension 0}}
                      {:op :stablehlo/broadcast_in_dim :invars [:iota_1d] :outvars [:iota_4d]
                       :attrs {:broadcast_dimensions [3] :target_shape [1 1 1 max-seq-len]}}
                      {:op :stablehlo/reshape :invars [:pos] :outvars [:pos_s] :attrs {:shape []}}
                      {:op :stablehlo/broadcast_in_dim :invars [:pos_s] :outvars [:pos_4d]
                       :attrs {:broadcast_dimensions [] :target_shape [1 1 1 max-seq-len]}}
                      {:op :stablehlo/compare :invars [:iota_4d :pos_4d] :outvars [:cmp_fut] :attrs {:comparison_direction "GT"}}
                      {:op :stablehlo/constant :value -10000.0 :type [:tensor [] :f32] :outvars [:c_neg]}
                      {:op :stablehlo/broadcast_in_dim :invars [:c_neg] :outvars [:c_neg_4d]
                       :attrs {:broadcast_dimensions [] :target_shape [1 1 1 max-seq-len]}}
                      {:op :stablehlo/constant :value 0.0 :type [:tensor [] :f32] :outvars [:c_zero]}
                      {:op :stablehlo/broadcast_in_dim :invars [:c_zero] :outvars [:c_zero_4d]
                       :attrs {:broadcast_dimensions [] :target_shape [1 1 1 max-seq-len]}}
                      {:op :stablehlo/select :invars [:cmp_fut :c_neg_4d :c_zero_4d] :outvars [:mask_4d]}
                      {:op :stablehlo/broadcast_in_dim :invars [:mask_4d] :outvars [:mask_bcast]
                       :attrs {:broadcast_dimensions [0 1 2 3] :target_shape [1 num-heads 1 max-seq-len]}}])
         eqs (vec (concat [{:op :stablehlo/broadcast_in_dim :invars [:k_cache] :outvars [:k_rep]
                            :attrs {:broadcast_dimensions [0 1 2 4] :target_shape [1 max-seq-len num-kv-heads group-size head-dim]}}
                           {:op :stablehlo/reshape :invars [:k_rep] :outvars [:k_heads]
                            :attrs {:shape [1 max-seq-len num-heads head-dim]}}
                           {:op :stablehlo/broadcast_in_dim :invars [:v_cache] :outvars [:v_rep]
                            :attrs {:broadcast_dimensions [0 1 2 4] :target_shape [1 max-seq-len num-kv-heads group-size head-dim]}}
                           {:op :stablehlo/reshape :invars [:v_rep] :outvars [:v_heads]
                            :attrs {:shape [1 max-seq-len num-heads head-dim]}}

                           {:op :stablehlo/dot_general :invars [:q_ro :k_heads] :outvars [:scores_4d]
                            :attrs {:batch_dims {:lhs [0 2] :rhs [0 2]} :contracting_dims {:lhs [3] :rhs [3]}}}
                           {:op :stablehlo/reshape :invars [:scores_4d] :outvars [:scores]
                            :attrs {:shape [1 num-heads 1 max-seq-len]}}]
                          mask-eqns
                          [{:op :stablehlo/add :invars [:scores :mask_bcast] :outvars [:masked_scores]}
                           {:op :stablehlo/reduce_max :invars [:masked_scores] :outvars [:max_val]
                            :attrs {:axes [-1] :keep_dims true}}
                           {:op :stablehlo/subtract :invars [:masked_scores :max_val] :outvars [:s_diff]}
                           {:op :stablehlo/exp :invars [:s_diff] :outvars [:s_exp]}
                           {:op :stablehlo/reduce_sum :invars [:s_exp] :outvars [:s_sum]
                            :attrs {:axes [-1] :keep_dims true}}
                           {:op :stablehlo/divide :invars [:s_exp :s_sum] :outvars [:probs]}

                           {:op :stablehlo/dot_general :invars [:probs :v_heads] :outvars [:ctx_raw]
                            :attrs {:batch_dims {:lhs [0 1] :rhs [0 2]} :contracting_dims {:lhs [3] :rhs [1]}}}
                           {:op :stablehlo/reshape :invars [:ctx_raw] :outvars [:ctx]
                            :attrs {:shape [1 1 num-heads head-dim]}}]))]
     (shlo/validate-graph {:name "test_standard_attn" :invars invars :outvars outvars :eqns eqs}))))

(defspec prop-chunked-attention-parity 10
  (prop/for-all [max-seq-len (gen/elements [64 128])
                 chunk-size (gen/elements [32 64])
                 num-heads (gen/elements [4 8])
                 num-kv-heads (gen/elements [1 2])
                 head-dim (gen/elements [32 64])
                 pos-idx (gen/choose 0 63)]
                (let [pos (min pos-idx (dec max-seq-len))
                      group-size (quot num-heads num-kv-heads)
                      num-heads (* num-kv-heads group-size)
                      invars [[:pos [:tensor [1] :i32]]
                              [:q_ro [:tensor [1 1 num-heads head-dim] :f32]]
                              [:k_cache [:tensor [1 max-seq-len num-kv-heads head-dim] :f32]]
                              [:v_cache [:tensor [1 max-seq-len num-kv-heads head-dim] :f32]]]
                      ast [:block {:name :chunked_block}
                           [:chunked-attention [:ctx :b :p :h :dh]
                            [:q_ro :b :p :h :dh]
                            [:k_cache :b :kv-s :kvh :dh]
                            [:v_cache :b :kv-s :kvh :dh]
                            {:pos :pos
                             :chunk-size chunk-size
                             :head-dim head-dim
                             :num-heads num-heads
                             :num-kv-heads num-kv-heads
                             :max-seq-len max-seq-len
                             :shape [1 1 num-heads head-dim]}]]
                      g-chunk (lower/ast->graph "test_chunk_parity" invars ast [:ctx])
                      g-std (test-standard-attention-graph max-seq-len num-heads num-kv-heads head-dim)
                      ctx (xla/init-cpu!)
                      exec-std (xla/compile-graph ctx g-std)
                      exec-chunk (xla/compile-graph ctx g-chunk)
                      pos-buf (xla/buffer-from-host-buffer (int-array [pos]) [1] 4)
                      q-data (float-array (* 1 1 num-heads head-dim))
                      _ (dotimes [i (count q-data)] (aset q-data i (float (Math/sin (double (inc i))))))
                      q-buf (xla/buffer-from-host-buffer q-data [1 1 num-heads head-dim] 11)
                      k-data (float-array (* 1 max-seq-len num-kv-heads head-dim))
                      _ (dotimes [i (count k-data)] (aset k-data i (float (Math/cos (double (inc i))))))
                      k-buf (xla/buffer-from-host-buffer k-data [1 max-seq-len num-kv-heads head-dim] 11)
                      v-data (float-array (* 1 max-seq-len num-kv-heads head-dim))
                      _ (dotimes [i (count v-data)] (aset v-data i (float (Math/sin (* 0.5 (double (inc i)))))))
                      v-buf (xla/buffer-from-host-buffer v-data [1 max-seq-len num-kv-heads head-dim] 11)
                      res-std (xla/execute exec-std [pos-buf q-buf k-buf v-buf])
                      res-chunk (xla/execute exec-chunk [pos-buf q-buf k-buf v-buf])
                      n-elem (* 1 1 num-heads head-dim)
                      arr-std (pjrt/buffer-to-host-buffer ctx res-std n-elem :f32)
                      arr-chunk (pjrt/buffer-to-host-buffer ctx res-chunk n-elem :f32)
                      max-diff (reduce max 0.0 (map #(Math/abs (double (- %1 %2))) arr-std arr-chunk))]
                  (< max-diff 1e-4))))

(defspec prop-sliding-window-ring-buffer-parity 10
  (prop/for-all [window (gen/elements [16 32])
                 mult (gen/elements [2 3])
                 chunk-size (gen/elements [8 16])
                 num-heads (gen/elements [4 8])
                 num-kv-heads (gen/elements [1 2])
                 head-dim (gen/elements [16 32])
                 pos-idx (gen/choose 0 95)]
                (let [max-seq-len (* window mult)
                      pos (min pos-idx (dec max-seq-len))
                      group-size (quot num-heads num-kv-heads)
                      num-heads (* num-kv-heads group-size)
                      g-std (test-standard-attention-graph max-seq-len num-heads num-kv-heads head-dim window)
                      invars-ring [[:pos [:tensor [1] :i32]]
                                   [:q_ro [:tensor [1 1 num-heads head-dim] :f32]]
                                   [:k_cache [:tensor [1 window num-kv-heads head-dim] :f32]]
                                   [:v_cache [:tensor [1 window num-kv-heads head-dim] :f32]]]
                      ast-ring [:block {:name :ring_block}
                                [:chunked-attention [:ctx :b :p :h :dh]
                                 [:q_ro :b :p :h :dh]
                                 [:k_cache :b :kv-s :kvh :dh]
                                 [:v_cache :b :kv-s :kvh :dh]
                                 {:pos :pos
                                  :chunk-size chunk-size
                                  :head-dim head-dim
                                  :num-heads num-heads
                                  :num-kv-heads num-kv-heads
                                  :max-seq-len window
                                  :sliding-window window
                                  :ring-buffer true
                                  :shape [1 1 num-heads head-dim]}]]
                      g-ring (lower/ast->graph "test_ring_parity" invars-ring ast-ring [:ctx])
                      ctx (xla/init-cpu!)
                      exec-std (xla/compile-graph ctx g-std)
                      exec-ring (xla/compile-graph ctx g-ring)
                      pos-buf (xla/buffer-from-host-buffer (int-array [pos]) [1] 4)
                      q-data (float-array (* 1 1 num-heads head-dim))
                      _ (dotimes [i (count q-data)] (aset q-data i (float (Math/sin (double (inc i))))))
                      q-buf (xla/buffer-from-host-buffer q-data [1 1 num-heads head-dim] 11)
                      k-full (float-array (* 1 max-seq-len num-kv-heads head-dim))
                      v-full (float-array (* 1 max-seq-len num-kv-heads head-dim))
                      _ (dotimes [i (count k-full)] (aset k-full i (float (Math/cos (double (inc i))))))
                      _ (dotimes [i (count v-full)] (aset v-full i (float (Math/sin (* 0.5 (double (inc i)))))))
                      k-buf-std (xla/buffer-from-host-buffer k-full [1 max-seq-len num-kv-heads head-dim] 11)
                      v-buf-std (xla/buffer-from-host-buffer v-full [1 max-seq-len num-kv-heads head-dim] 11)
                      tok-stride (* num-kv-heads head-dim)
                      k-ring (float-array (* 1 window num-kv-heads head-dim))
                      v-ring (float-array (* 1 window num-kv-heads head-dim))
                      _ (dotimes [t (inc pos)]
                          (let [slot (mod t window)
                                src-offset (* t tok-stride)
                                dst-offset (* slot tok-stride)]
                            (System/arraycopy k-full src-offset k-ring dst-offset tok-stride)
                            (System/arraycopy v-full src-offset v-ring dst-offset tok-stride)))
                      k-buf-ring (xla/buffer-from-host-buffer k-ring [1 window num-kv-heads head-dim] 11)
                      v-buf-ring (xla/buffer-from-host-buffer v-ring [1 window num-kv-heads head-dim] 11)
                      res-std (xla/execute exec-std [pos-buf q-buf k-buf-std v-buf-std])
                      res-ring (xla/execute exec-ring [pos-buf q-buf k-buf-ring v-buf-ring])
                      n-elem (* 1 1 num-heads head-dim)
                      arr-std (pjrt/buffer-to-host-buffer ctx res-std n-elem :f32)
                      arr-ring (pjrt/buffer-to-host-buffer ctx res-ring n-elem :f32)
                      max-diff (reduce max 0.0 (map #(Math/abs (double (- %1 %2))) arr-std arr-ring))]
                  (< max-diff 1e-4))))

