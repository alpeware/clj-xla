(ns clj-xla.models.gemma
  "Gemma 2B / Gemma 2 Architecture Assembler and Safetensors Parameter Key Mapping."
  (:refer-clojure :exclude [+ * /])
  (:require [clj-xla.nn.activations :refer [geglu]]
            [clj-xla.nn.attention :refer [apply-rope gqa-causal-attention linear]]
            [clj-xla.nn.norm :refer [gemma-rms-norm]]
            [clj-xla.tensor :refer [* + / gather reshape tanh transpose]]))

(def DEFAULT_GEMMA_CONFIG
  {:vocab-size 256000
   :hidden-size 2048
   :intermediate-size 16384
   :num-hidden-layers 18
   :num-attention-heads 8
   :num-key-value-heads 1
   :head-dim 256
   :rms-norm-eps 1e-6})

(defn gemma-config
  "Returns Gemma configuration map with optional custom overrides."
  ([] DEFAULT_GEMMA_CONFIG)
  ([overrides] (merge DEFAULT_GEMMA_CONFIG overrides)))

(defn- embed-lookup [x embed-tokens hidden-dim]
  (let [raw-embed (gather embed-tokens x)
        scale (Math/sqrt (double hidden-dim))]
    (* raw-embed scale)))

(defn gemma-mlp
  "Gemma GeGLU MLP Block: down_proj(gelu(gate_proj(x)) * up_proj(x))."
  [x gate-w up-w down-w]
  (geglu x gate-w up-w down-w))

(defn gemma-attention
  "Gemma Multi-Head / Grouped-Query Causal Self-Attention with RoPE."
  ([x q-w k-w v-w o-w num-heads num-kv-heads pos-ids]
   (gemma-attention x q-w k-w v-w o-w num-heads num-kv-heads pos-ids nil nil))
  ([x q-w k-w v-w o-w num-heads num-kv-heads pos-ids past-kv pos]
   (let [q-w-t (transpose q-w [1 0])
         k-w-t (transpose k-w [1 0])
         v-w-t (transpose v-w [1 0])
         q (linear x q-w-t nil)
         k (linear x k-w-t nil)
         v (linear x v-w-t nil)
         [_ [batch seq-len q-dim] _] (:type q)
         [_ [_ _ kv-dim] _] (:type k)
         head-dim (quot q-dim num-heads)
         q-4d (transpose (reshape q [batch seq-len num-heads head-dim]) [0 2 1 3])
         k-4d (transpose (reshape k [batch seq-len num-kv-heads head-dim]) [0 2 1 3])
         [q-rope k-rope] (apply-rope q-4d k-4d pos-ids)
         q-rope-3d (reshape (transpose q-rope [0 2 1 3]) [batch seq-len q-dim])
         k-rope-3d (reshape (transpose k-rope [0 2 1 3]) [batch seq-len kv-dim])]
     (if (some? past-kv)
       (gqa-causal-attention q-rope-3d k-rope-3d v o-w num-heads num-kv-heads 50.0 past-kv pos)
       (gqa-causal-attention q-rope-3d k-rope-3d v o-w num-heads num-kv-heads 50.0)))))

(defn gemma-block
  "Single Gemma 2 Transformer block with 4 RMSNorms and dual residual connections."
  ([x weights num-heads num-kv-heads pos-ids]
   (gemma-block x weights num-heads num-kv-heads pos-ids nil nil))
  ([x weights num-heads num-kv-heads pos-ids past-kv pos]
   (let [{:keys [input-ln-w q-w k-w v-w o-w
                 post-attn-ln-w pre-mlp-ln-w post-mlp-ln-w
                 gate-w up-w down-w]} weights
         ;; Attention Sub-block
         x-norm1 (gemma-rms-norm x input-ln-w 1e-6)
         attn-res (if (some? past-kv)
                    (gemma-attention x-norm1 q-w k-w v-w o-w num-heads num-kv-heads pos-ids past-kv pos)
                    (gemma-attention x-norm1 q-w k-w v-w o-w num-heads num-kv-heads pos-ids))
         [attn-raw updated-kv] (if (vector? attn-res) attn-res [attn-res nil])
         attn-normed (if post-attn-ln-w (gemma-rms-norm attn-raw post-attn-ln-w 1e-6) attn-raw)
         x-res1 (+ x attn-normed)

         ;; MLP Sub-block
         x-norm2 (if pre-mlp-ln-w (gemma-rms-norm x-res1 pre-mlp-ln-w 1e-6) (gemma-rms-norm x-res1 post-attn-ln-w 1e-6))
         mlp-raw (gemma-mlp x-norm2 gate-w up-w down-w)
         mlp-normed (if post-mlp-ln-w (gemma-rms-norm mlp-raw post-mlp-ln-w 1e-6) mlp-raw)
         x-out (+ x-res1 mlp-normed)]
     (if (some? past-kv)
       [x-out updated-kv]
       x-out))))

(defn full-gemma-forward
  "Full Gemma Transformer forward pass."
  ([x embed-tokens layers-weights final-norm-w pos-ids]
   (full-gemma-forward x embed-tokens layers-weights final-norm-w pos-ids nil nil))
  ([x embed-tokens layers-weights final-norm-w pos-ids arg6 arg7]
   (if (number? arg6)
     (full-gemma-forward x embed-tokens layers-weights final-norm-w pos-ids arg6 arg7 nil nil)
     (let [first-layer (first layers-weights)
           [_ [q-dim _] _] (:type (:q-w first-layer))
           [_ [kv-dim _] _] (:type (:k-w first-layer))
           num-heads (quot q-dim 256)
           num-kv-heads (quot kv-dim 256)]
       (full-gemma-forward x embed-tokens layers-weights final-norm-w pos-ids num-heads num-kv-heads arg6 arg7))))
  ([x embed-tokens layers-weights final-norm-w pos-ids num-heads num-kv-heads kv-caches pos]
   (let [[_ [_vocab-size hidden-dim] _] (:type embed-tokens)
         tok-embed (embed-lookup x embed-tokens hidden-dim)
         use-kv? (some? kv-caches)
         [x-out new-kv-caches]
         (if use-kv?
           (reduce (fn [[h updated-acc] [layer-w layer-kv]]
                     (let [[h-next new-kv] (gemma-block h layer-w num-heads num-kv-heads pos-ids layer-kv pos)]
                       [h-next (conj updated-acc new-kv)]))
                   [tok-embed []]
                   (map vector layers-weights kv-caches))
           [(reduce (fn [h layer-w]
                      (gemma-block h layer-w num-heads num-kv-heads pos-ids))
                    tok-embed
                    layers-weights)
            nil])
         normed (gemma-rms-norm x-out final-norm-w 1e-6)
         embed-t (transpose embed-tokens [1 0])
         raw-logits (linear normed embed-t nil)
         capped-logits (* 30.0 (tanh (/ raw-logits 30.0)))]
     (if use-kv?
       [capped-logits new-kv-caches]
       capped-logits))))



(defn weight-key-map [layer-idx]
  (let [prefix (str "model.layers." layer-idx ".")]
    {:input-ln-w     (str prefix "input_layernorm.weight")
     :q-w            (str prefix "self_attn.q_proj.weight")
     :k-w            (str prefix "self_attn.k_proj.weight")
     :v-w            (str prefix "self_attn.v_proj.weight")
     :o-w            (str prefix "self_attn.o_proj.weight")
     :post-attn-ln-w (str prefix "post_attention_layernorm.weight")
     :pre-mlp-ln-w   (str prefix "pre_feedforward_layernorm.weight")
     :post-mlp-ln-w  (str prefix "post_feedforward_layernorm.weight")
     :gate-w         (str prefix "mlp.gate_proj.weight")
     :up-w           (str prefix "mlp.up_proj.weight")
     :down-w         (str prefix "mlp.down_proj.weight")}))
