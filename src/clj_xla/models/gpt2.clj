(ns clj-xla.models.gpt2
  "GPT-2 Architecture Assembler and Safetensors Parameter Key Mapping."
  (:refer-clojure :exclude [+])
  (:require [clj-xla.nn.activations :refer [gelu]]
            [clj-xla.nn.attention :refer [causal-self-attention linear]]
            [clj-xla.nn.norm :refer [layer-norm]]
            [clj-xla.tensor :refer [+]]))

(def DEFAULT_GPT2_CONFIG
  {:vocab-size 50257
   :n-positions 1024
   :n-embd 768
   :n-layer 12
   :n-head 12
   :layer-norm-epsilon 1e-5})

(defn gpt2-config
  "Returns GPT-2 configuration map with optional custom overrides."
  ([] DEFAULT_GPT2_CONFIG)
  ([overrides] (merge DEFAULT_GPT2_CONFIG overrides)))

(defn gpt2-mlp
  "GPT-2 Feed-Forward MLP Block: GELU(x @ fc_w + fc_b) @ proj_w + proj_b."
  [x fc-w fc-b proj-w proj-b]
  (let [h (gelu (linear x fc-w fc-b))]
    (linear h proj-w proj-b)))

(defn gpt2-block
  "Single GPT-2 Transformer layer block with pre-layer normalization and residual connections."
  [x weights num-heads]
  (let [{:keys [ln1-g ln1-b c-attn-w c-attn-b c-proj-w c-proj-b
                ln2-g ln2-b mlp-fc-w mlp-fc-b mlp-proj-w mlp-proj-b]} weights
        x-norm1 (layer-norm x ln1-g ln1-b)
        attn-out (causal-self-attention x-norm1 c-attn-w c-attn-b c-proj-w c-proj-b num-heads)
        x-res1 (+ x attn-out)
        x-norm2 (layer-norm x-res1 ln2-g ln2-b)
        mlp-out (gpt2-mlp x-norm2 mlp-fc-w mlp-fc-b mlp-proj-w mlp-proj-b)]
    (+ x-res1 mlp-out)))

(defn full-gpt2-forward
  "Full GPT-2 Transformer forward pass: multi-block sequence -> final LayerNorm -> LM head vocabulary projection."
  [x layers-weights ln-f-g ln-f-b lm-head-w]
  (let [x-out (reduce (fn [h layer-w]
                        (gpt2-block h layer-w 12))
                      x
                      layers-weights)
        normed (layer-norm x-out ln-f-g ln-f-b)
        logits (linear normed lm-head-w nil)]
    logits))

(defn weight-key-map
  "Maps HuggingFace GPT-2 Safetensors tensor names to internal key names for layer `layer-idx`."
  [layer-idx]
  (let [prefix (str "h." layer-idx ".")]
    {:ln1-g (str prefix "ln_1.weight")
     :ln1-b (str prefix "ln_1.bias")
     :c-attn-w (str prefix "attn.c_attn.weight")
     :c-attn-b (str prefix "attn.c_attn.bias")
     :c-proj-w (str prefix "attn.c_proj.weight")
     :c-proj-b (str prefix "attn.c_proj.bias")
     :ln2-g (str prefix "ln_2.weight")
     :ln2-b (str prefix "ln_2.bias")
     :mlp-fc-w (str prefix "mlp.c_fc.weight")
     :mlp-fc-b (str prefix "mlp.c_fc.bias")
     :mlp-proj-w (str prefix "mlp.c_proj.weight")
     :mlp-proj-b (str prefix "mlp.c_proj.bias")}))
