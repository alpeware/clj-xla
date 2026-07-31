(ns clj-xla.models.smollm
  "SmolLM Architecture Assembler and Safetensors Parameter Key Mapping."
  (:refer-clojure :exclude [+])
  (:require [clj-xla.nn.activations :refer [swiglu]]
            [clj-xla.nn.attention :refer [apply-rope gqa-causal-attention linear]]
            [clj-xla.nn.norm :refer [rms-norm]]
            [clj-xla.tensor :refer [+]]))

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
  (swiglu x gate-w up-w down-w))

(defn smollm-attention
  "SmolLM Multi-Head Causal Self-Attention with RoPE and Grouped-Query Attention."
  [x q-w k-w v-w o-w num-heads num-kv-heads pos-ids]
  (let [q (linear x q-w nil)
        k (linear x k-w nil)
        v (linear x v-w nil)
        [q-rope k-rope] (apply-rope q k pos-ids)]
    (gqa-causal-attention q-rope k-rope v o-w num-heads num-kv-heads)))

(defn smollm-block
  "Single SmolLM Llama-style Transformer layer block."
  [x weights num-heads num-kv-heads pos-ids]
  (let [{:keys [input-ln-w q-w k-w v-w o-w
                post-attn-ln-w gate-w up-w down-w]} weights
        x-norm1 (rms-norm x input-ln-w 1e-5)
        attn-out (smollm-attention x-norm1 q-w k-w v-w o-w num-heads num-kv-heads pos-ids)
        x-res1 (+ x attn-out)
        x-norm2 (rms-norm x-res1 post-attn-ln-w 1e-5)
        mlp-out (smollm-mlp x-norm2 gate-w up-w down-w)]
    (+ x-res1 mlp-out)))

(defn full-smollm-forward
  "Full SmolLM Transformer forward pass: multi-block sequence -> final RMSNorm -> LM head vocabulary projection."
  [x layers-weights final-norm-w lm-head-w pos-ids]
  (let [x-out (reduce (fn [h layer-w]
                        (smollm-block h layer-w 9 3 pos-ids))
                      x
                      layers-weights)
        normed (rms-norm x-out final-norm-w 1e-5)
        logits (linear normed lm-head-w nil)]
    logits))

(defn weight-key-map
  "Maps HuggingFace SmolLM Safetensors tensor names to internal key names for layer `layer-idx`."
  [layer-idx]
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
