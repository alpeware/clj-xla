(ns clj-xla.models.gpt2
  "GPT-2 Architecture Assembler and Safetensors Parameter Key Mapping."
  (:refer-clojure :exclude [+])
  (:require [clj-xla.nn.activations :refer [gelu]]
            [clj-xla.nn.attention :refer [causal-self-attention linear]]
            [clj-xla.nn.norm :refer [layer-norm]]
            [clj-xla.tensor :refer [+]])
  (:import [clj_xla Gpt2FastEngine]))

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

(defn eval-gpt2-sequence
  "Evaluates 12-layer GPT-2 Transformer forward pass over sequence X using SIMD-vectorized Java 25 engine.
   Returns final LayerNorm hidden state float-array of length 768 for the last token position S-1."
  [X layers-weights ^floats ln-f-g ^floats ln-f-b]
  (let [float-array-type (type (float-array 0))
        ^"[[F" X-arr (into-array float-array-type X)
        ^"[[F" ln1g (into-array float-array-type (mapv :ln1-g layers-weights))
        ^"[[F" ln1b (into-array float-array-type (mapv :ln1-b layers-weights))
        ^"[[F" cAtW (into-array float-array-type (mapv :c-attn-w layers-weights))
        ^"[[F" cAtB (into-array float-array-type (mapv :c-attn-b layers-weights))
        ^"[[F" cPrW (into-array float-array-type (mapv :c-proj-w layers-weights))
        ^"[[F" cPrB (into-array float-array-type (mapv :c-proj-b layers-weights))
        ^"[[F" ln2g (into-array float-array-type (mapv :ln2-g layers-weights))
        ^"[[F" ln2b (into-array float-array-type (mapv :ln2-b layers-weights))
        ^"[[F" fcW (into-array float-array-type (mapv :mlp-fc-w layers-weights))
        ^"[[F" fcB (into-array float-array-type (mapv :mlp-fc-b layers-weights))
        ^"[[F" prW (into-array float-array-type (mapv :mlp-proj-w layers-weights))
        ^"[[F" prB (into-array float-array-type (mapv :mlp-proj-b layers-weights))]
    (Gpt2FastEngine/evalSequence X-arr ln1g ln1b cAtW cAtB cPrW cPrB ln2g ln2b fcW fcB prW prB ln-f-g ln-f-b)))

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
