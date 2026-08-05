(ns clj-xla.models.smollm
  "SmolLM Architecture Assembler and Safetensors Parameter Key Mapping."
  (:refer-clojure :exclude [+])
  (:require [clj-xla.nn.activations :refer [swiglu]]
            [clj-xla.nn.attention :refer [apply-rope gqa-causal-attention linear]]
            [clj-xla.nn.norm :refer [rms-norm]]
            [clj-xla.tensor :refer [+ gather transpose]]))

(def DEFAULT_SMOLLM_CONFIG
  {:vocab-size 49152
   :n-positions 2048
   :n-embd 576
   :intermediate-size 1536
   :n-layer 30
   :n-head 9
   :n-kv-head 3
   :rms-norm-eps 1e-5})

(defn smollm-config
  "Returns SmolLM configuration map with optional custom overrides."
  ([] DEFAULT_SMOLLM_CONFIG)
  ([overrides] (merge DEFAULT_SMOLLM_CONFIG overrides)))

(defn smollm-mlp
  "SmolLM SwiGLU MLP Block: down_proj(silu(gate_proj(x)) * up_proj(x))."
  [x gate-w up-w down-w]
  (let [gate-w-t (transpose gate-w [1 0])
        up-w-t (transpose up-w [1 0])
        down-w-t (transpose down-w [1 0])]
    (swiglu x gate-w-t up-w-t down-w-t)))

(defn smollm-attention
  "SmolLM Multi-Head Causal Self-Attention with RoPE and Grouped-Query Attention."
  ([x q-w k-w v-w o-w num-heads num-kv-heads pos-ids]
   (smollm-attention x q-w k-w v-w o-w num-heads num-kv-heads pos-ids nil nil))
  ([x q-w k-w v-w o-w num-heads num-kv-heads pos-ids past-kv pos]
   (let [q-w-t (transpose q-w [1 0])
         k-w-t (transpose k-w [1 0])
         v-w-t (transpose v-w [1 0])
         q (linear x q-w-t nil)
         k (linear x k-w-t nil)
         v (linear x v-w-t nil)
         [q-rope k-rope] (apply-rope q k pos-ids)]
     (if (some? past-kv)
       (gqa-causal-attention q-rope k-rope v o-w num-heads num-kv-heads 50.0 past-kv pos)
       (gqa-causal-attention q-rope k-rope v o-w num-heads num-kv-heads)))))

(defn smollm-block
  "Single SmolLM Llama-style Transformer layer block."
  ([x weights num-heads num-kv-heads pos-ids]
   (smollm-block x weights num-heads num-kv-heads pos-ids nil nil))
  ([x weights num-heads num-kv-heads pos-ids past-kv pos]
   (let [{:keys [input-ln-w q-w k-w v-w o-w
                 post-attn-ln-w gate-w up-w down-w]} weights
         x-norm1 (rms-norm x input-ln-w 1e-5)
         attn-res (if (some? past-kv)
                    (smollm-attention x-norm1 q-w k-w v-w o-w num-heads num-kv-heads pos-ids past-kv pos)
                    (smollm-attention x-norm1 q-w k-w v-w o-w num-heads num-kv-heads pos-ids))
         [attn-out updated-kv] (if (vector? attn-res) attn-res [attn-res nil])
         x-res1 (+ x attn-out)
         x-norm2 (rms-norm x-res1 post-attn-ln-w 1e-5)
         mlp-out (smollm-mlp x-norm2 gate-w up-w down-w)
         x-out (+ x-res1 mlp-out)]
     (if (some? past-kv)
       [x-out updated-kv]
       x-out))))

(defn full-smollm-forward
  "Full SmolLM Transformer forward pass: multi-block sequence -> final RMSNorm -> LM head vocabulary projection."
  ([x embed-tokens layers-weights final-norm-w lm-head-w pos-ids]
   (full-smollm-forward x embed-tokens layers-weights final-norm-w lm-head-w pos-ids nil nil))
  ([x embed-tokens layers-weights final-norm-w lm-head-w pos-ids kv-caches pos]
   (let [tok-embed (gather embed-tokens x)
         use-kv? (some? kv-caches)
         [x-out new-kv-caches]
         (if use-kv?
           (reduce (fn [[h updated-acc] [layer-w layer-kv]]
                     (let [[h-next new-kv] (smollm-block h layer-w 9 3 pos-ids layer-kv pos)]
                       [h-next (conj updated-acc new-kv)]))
                   [tok-embed []]
                   (map vector layers-weights kv-caches))
           [(reduce (fn [h layer-w]
                      (smollm-block h layer-w 9 3 pos-ids))
                    tok-embed
                    layers-weights)
            nil])
         normed (rms-norm x-out final-norm-w 1e-5)
         lm-head-t (transpose lm-head-w [1 0])
         logits (linear normed lm-head-t nil)]
     (if use-kv?
       [logits new-kv-caches]
       logits))))

(defn weight-key-map [layer-idx]
  (let [prefix (str "model.layers." layer-idx ".")]
    {:input-ln-w (str prefix "input_layernorm.weight")
     :q-w (str prefix "self_attn.q_proj.weight")
     :k-w (str prefix "self_attn.k_proj.weight")
     :v-w (str prefix "self_attn.v_proj.weight")
     :o-w (str prefix "self_attn.o_proj.weight")
     :post-attn-ln-w (str prefix "post_attention_layernorm.weight")
     :gate-w (str prefix "mlp.gate_proj.weight")
     :up-w (str prefix "mlp.up_proj.weight")
     :down-w (str prefix "mlp.down_proj.weight")}))
