(ns clj-xla.models.gemma
  "Gemma 1 / Gemma 2 / Gemma 3 Architecture Assembler and Safetensors Parameter Key Mapping."
  (:refer-clojure :exclude [+ * /])
  (:require [clj-xla.nn.activations :refer [geglu]]
            [clj-xla.nn.attention :refer [apply-rope gqa-causal-attention linear]]
            [clj-xla.nn.norm :refer [gemma-rms-norm rms-norm]]
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

(def DEFAULT_GEMMA3_270M_CONFIG
  {:vocab-size 262144
   :hidden-size 640
   :intermediate-size 2048
   :num-hidden-layers 18
   :num-attention-heads 4
   :num-key-value-heads 1
   :head-dim 256
   :rms-norm-eps 1e-6
   :query-pre-attn-scalar 256
   :sliding-window 512
   :rope-theta 1000000.0
   :rope-local-base-freq 10000.0
   :layer-types ["sliding_attention" "sliding_attention" "sliding_attention" "sliding_attention" "sliding_attention" "full_attention"
                 "sliding_attention" "sliding_attention" "sliding_attention" "sliding_attention" "sliding_attention" "full_attention"
                 "sliding_attention" "sliding_attention" "sliding_attention" "sliding_attention" "sliding_attention" "full_attention"]})

(def DEFAULT_GEMMA4_E2B_CONFIG
  {:vocab-size 262144
   :hidden-size 1536
   :intermediate-size 6144
   :num-hidden-layers 35
   :num-attention-heads 8
   :num-key-value-heads 1
   :head-dim 256
   :rms-norm-eps 1e-6
   :hidden-size-per-layer-input 256
   :num-kv-shared-layers 20
   :sliding-window 512
   :rope-parameters {:full-attention {:rope-theta 1000000.0 :partial-rotary-factor 0.25}
                     :sliding-attention {:rope-theta 10000.0}}
   :final-logit-softcap 30.0
   :layer-types ["sliding_attention" "sliding_attention" "sliding_attention" "sliding_attention" "full_attention"
                 "sliding_attention" "sliding_attention" "sliding_attention" "sliding_attention" "full_attention"
                 "sliding_attention" "sliding_attention" "sliding_attention" "sliding_attention" "full_attention"
                 "sliding_attention" "sliding_attention" "sliding_attention" "sliding_attention" "full_attention"
                 "sliding_attention" "sliding_attention" "sliding_attention" "sliding_attention" "full_attention"
                 "sliding_attention" "sliding_attention" "sliding_attention" "sliding_attention" "full_attention"
                 "sliding_attention" "sliding_attention" "sliding_attention" "sliding_attention" "full_attention"]})

(defn gemma-config
  "Returns Gemma 1/2 configuration map with optional custom overrides."
  ([] DEFAULT_GEMMA_CONFIG)
  ([overrides] (merge DEFAULT_GEMMA_CONFIG overrides)))

(defn gemma3-config
  "Returns Gemma 3 configuration map with optional custom overrides."
  ([] DEFAULT_GEMMA3_270M_CONFIG)
  ([overrides] (merge DEFAULT_GEMMA3_270M_CONFIG overrides)))

(defn gemma4-config
  "Returns Gemma 4 configuration map with optional custom overrides."
  ([] DEFAULT_GEMMA4_E2B_CONFIG)
  ([overrides] (merge DEFAULT_GEMMA4_E2B_CONFIG overrides)))

(defn embed-lookup [x embed-tokens hidden-dim]
  (let [raw-embed (gather embed-tokens x)
        scale (Math/sqrt (double hidden-dim))]
    (* raw-embed scale)))

(defn gemma-mlp
  "Gemma GeGLU MLP Block: down_proj(gelu(gate_proj(x)) * up_proj(x))."
  [x gate-w up-w down-w]
  (geglu x gate-w up-w down-w))

(defn gemma-attention
  "Gemma Multi-Head / Grouped-Query Causal Self-Attention with RoPE and optional QK Norm."
  ([x q-w k-w v-w o-w num-heads num-kv-heads pos-ids]
   (gemma-attention x q-w k-w v-w o-w num-heads num-kv-heads pos-ids nil nil {}))
  ([x q-w k-w v-w o-w num-heads num-kv-heads pos-ids past-kv pos]
   (gemma-attention x q-w k-w v-w o-w num-heads num-kv-heads pos-ids past-kv pos {}))
  ([x q-w k-w v-w o-w num-heads num-kv-heads pos-ids past-kv pos opts]
   (let [{:keys [q-norm-w k-norm-w theta-base attn-softcap rotary-dim norm-fn rope-proportion]
          :or {theta-base 10000.0 attn-softcap nil norm-fn gemma-rms-norm rope-proportion 1.0}} opts
         q-w-t (transpose q-w [1 0])
         k-w-t (transpose k-w [1 0])
         v-w-t (transpose v-w [1 0])
         q (linear x q-w-t nil)
         k (linear x k-w-t nil)
         v-raw (linear x v-w-t nil)
         v (if (some? q-norm-w) (rms-norm v-raw nil 1e-6) v-raw)
         [_ [batch seq-len q-dim] _] (:type q)
         [_ [_ _ kv-dim] _] (:type k)
         head-dim (quot q-dim num-heads)
         q-4d (transpose (reshape q [batch seq-len num-heads head-dim]) [0 2 1 3])
         k-4d (transpose (reshape k [batch seq-len num-kv-heads head-dim]) [0 2 1 3])
         ;; Apply QK Norm if weights provided (Gemma 3 / 4)
         q-normed (if (some? q-norm-w) (norm-fn q-4d q-norm-w 1e-6) q-4d)
         k-normed (if (some? k-norm-w) (norm-fn k-4d k-norm-w 1e-6) k-4d)
         ;; Apply RoPE
         [q-rope k-rope] (apply-rope q-normed k-normed pos-ids head-dim theta-base rotary-dim nil rope-proportion)
         q-rope-3d (reshape (transpose q-rope [0 2 1 3]) [batch seq-len q-dim])
         k-rope-3d (reshape (transpose k-rope [0 2 1 3]) [batch seq-len kv-dim])
         attn-scale (get opts :scale (if (some? q-norm-w) 1.0 (/ 1.0 (Math/sqrt (double head-dim)))))
         attn-opts {:scale attn-scale}]
     (if (some? past-kv)
       (gqa-causal-attention q-rope-3d k-rope-3d v o-w num-heads num-kv-heads attn-softcap past-kv pos attn-opts)
       (gqa-causal-attention q-rope-3d k-rope-3d v o-w num-heads num-kv-heads attn-softcap nil nil attn-opts)))))

(defn gemma-block
  "Single Gemma 1/2/3/4 Transformer block with RMSNorms, optional QK Norm, and dual residual connections."
  ([x weights num-heads num-kv-heads pos-ids]
   (gemma-block x weights num-heads num-kv-heads pos-ids nil nil))
  ([x weights num-heads num-kv-heads pos-ids past-kv pos]
   (let [{:keys [input-ln-w q-w k-w v-w o-w q-norm-w k-norm-w
                 post-attn-ln-w pre-mlp-ln-w post-mlp-ln-w
                 gate-w up-w down-w theta-base attn-softcap rotary-dim rope-proportion
                 per-layer-gate-w per-layer-proj-w post-per-layer-norm-w
                 per-layer-input layer-scalar-w]
          :or {theta-base 10000.0 attn-softcap nil rope-proportion 1.0}} weights
         norm-fn (or (:norm-fn weights) gemma-rms-norm)
         ;; 1. Attention Sub-block
         x-norm1 (norm-fn x input-ln-w 1e-6)
         attn-opts {:q-norm-w q-norm-w
                    :k-norm-w k-norm-w
                    :theta-base theta-base
                    :attn-softcap attn-softcap
                    :rotary-dim rotary-dim
                    :rope-proportion rope-proportion
                    :norm-fn norm-fn}
         attn-res (if (some? past-kv)
                    (gemma-attention x-norm1 q-w k-w v-w o-w num-heads num-kv-heads pos-ids past-kv pos attn-opts)
                    (gemma-attention x-norm1 q-w k-w v-w o-w num-heads num-kv-heads pos-ids nil nil attn-opts))
         [attn-raw updated-kv] (if (vector? attn-res) attn-res [attn-res nil])
         attn-normed (if post-attn-ln-w (norm-fn attn-raw post-attn-ln-w 1e-6) attn-raw)
         attn-out (+ attn-normed x)

         ;; 2. MLP Sub-block
         x-norm2 (if (some? pre-mlp-ln-w) (norm-fn attn-out pre-mlp-ln-w 1e-6) attn-out)
         mlp-raw (gemma-mlp x-norm2 gate-w up-w down-w)
         mlp-normed (if post-mlp-ln-w (norm-fn mlp-raw post-mlp-ln-w 1e-6) mlp-raw)
         mlp-out (+ mlp-normed attn-out)

         ;; 3. Gemma 4 Per-Layer Input (PLE) Gating Sub-block
         ple-out (if (and (some? per-layer-gate-w) (some? per-layer-proj-w) (some? per-layer-input))
                   (let [pl-gate (transpose per-layer-gate-w [1 0])
                         pl-proj (transpose per-layer-proj-w [1 0])
                         gate-raw (linear mlp-out pl-gate nil)
                         act-gate (clj-xla.nn.activations/gelu gate-raw)
                         gated (* act-gate per-layer-input)
                         proj-raw (linear gated pl-proj nil)
                         normed (if post-per-layer-norm-w (norm-fn proj-raw post-per-layer-norm-w 1e-6) proj-raw)]
                     (+ mlp-out normed))
                   mlp-out)

         ;; 4. Gemma 4 Layer Scalar (skip scale)
         x-out (if (some? layer-scalar-w)
                 (* ple-out layer-scalar-w)
                 ple-out)]
     (if (some? past-kv)
       [x-out updated-kv]
       x-out))))

(defn full-gemma-forward
  "Full Gemma (1 / 2 / 3) Transformer forward pass."
  ([x embed-tokens layers-weights final-norm-w pos-ids]
   (full-gemma-forward x embed-tokens layers-weights final-norm-w pos-ids nil nil {}))
  ([x embed-tokens layers-weights final-norm-w pos-ids arg6 arg7]
   (if (number? arg6)
     (full-gemma-forward x embed-tokens layers-weights final-norm-w pos-ids arg6 arg7 nil nil {})
     (full-gemma-forward x embed-tokens layers-weights final-norm-w pos-ids arg6 arg7 {})))
  ([x embed-tokens layers-weights final-norm-w pos-ids arg6 arg7 arg8]
   (if (number? arg6)
     (full-gemma-forward x embed-tokens layers-weights final-norm-w pos-ids arg6 arg7 arg8 nil {})
     (let [first-layer (first layers-weights)
           [_ [q-dim _] _] (:type (:q-w first-layer))
           [_ [kv-dim _] _] (:type (:k-w first-layer))
           num-heads (quot q-dim 256)
           num-kv-heads (quot kv-dim 256)]
       (full-gemma-forward x embed-tokens layers-weights final-norm-w pos-ids num-heads num-kv-heads arg6 arg7))))
  ([x embed-tokens layers-weights final-norm-w pos-ids num-heads num-kv-heads kv-caches pos]
   (full-gemma-forward x embed-tokens layers-weights final-norm-w pos-ids num-heads num-kv-heads kv-caches pos {}))
  ([x embed-tokens layers-weights final-norm-w pos-ids num-heads num-kv-heads kv-caches pos opts]
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
         final-softcap (get opts :final-logit-softcap 30.0)
         capped-logits (if (and (number? final-softcap) (pos? final-softcap))
                         (* final-softcap (tanh (/ raw-logits final-softcap)))
                         raw-logits)]
     (if use-kv?
       [capped-logits new-kv-caches]
       capped-logits))))

(defn full-gemma4-forward
  "Full Gemma 4 Transformer forward pass including Per-Layer Embeddings (PLE) and Layer Scalars."
  ([x embed-tokens embed-per-layer per-layer-model-proj-w per-layer-proj-norm-w layers-weights final-norm-w pos-ids]
   (full-gemma4-forward x embed-tokens embed-per-layer per-layer-model-proj-w per-layer-proj-norm-w layers-weights final-norm-w pos-ids 8 1 nil nil {}))
  ([x embed-tokens embed-per-layer per-layer-model-proj-w per-layer-proj-norm-w layers-weights final-norm-w pos-ids num-heads num-kv-heads]
   (full-gemma4-forward x embed-tokens embed-per-layer per-layer-model-proj-w per-layer-proj-norm-w layers-weights final-norm-w pos-ids num-heads num-kv-heads nil nil {}))
  ([x embed-tokens embed-per-layer per-layer-model-proj-w per-layer-proj-norm-w layers-weights final-norm-w pos-ids num-heads num-kv-heads kv-caches pos]
   (full-gemma4-forward x embed-tokens embed-per-layer per-layer-model-proj-w per-layer-proj-norm-w layers-weights final-norm-w pos-ids num-heads num-kv-heads kv-caches pos {}))
  ([x embed-tokens embed-per-layer per-layer-model-proj-w per-layer-proj-norm-w layers-weights final-norm-w pos-ids num-heads num-kv-heads kv-caches pos opts]
   (let [[_ [batch seq-len] _] (:type x)
         [_ [_vocab-size hidden-dim] _] (:type embed-tokens)
         num-layers (count layers-weights)
         tok-embed (embed-lookup x embed-tokens hidden-dim)
         ;; Compute PLE (Per-Layer Embedding) representation
         raw-pl-tok (gather embed-per-layer x)
         [_ [_pl_v total-pl-dim] _] (:type embed-per-layer)
         pl-dim (quot total-pl-dim num-layers)
         pl-tok-scaled (* raw-pl-tok (Math/sqrt (double pl-dim)))
         pl-proj-t (transpose per-layer-model-proj-w [1 0])
         pl-context-raw (linear tok-embed pl-proj-t nil)
         pl-tok-4d (reshape pl-tok-scaled [batch seq-len num-layers pl-dim])
         pl-context-4d (reshape pl-context-raw [batch seq-len num-layers pl-dim])
         pl-context-norm (rms-norm pl-context-4d per-layer-proj-norm-w 1e-6)
         ple-all (* (+ pl-context-norm pl-tok-4d) (/ 1.0 (Math/sqrt 2.0)))

         layers-with-ple (mapv (fn [i lw]
                                 (let [pl-slice (clj-xla.tensor/slice ple-all [0 0 i 0] [batch seq-len (inc i) pl-dim] [1 1 1 1])
                                       pl-input (reshape pl-slice [batch seq-len pl-dim])]
                                   (assoc lw :per-layer-input pl-input :norm-fn rms-norm)))
                               (range num-layers)
                               layers-weights)
         use-kv? (some? kv-caches)
         [x-out new-kv-caches]
         (if use-kv?
           (reduce (fn [[h updated-acc] [layer-w layer-kv]]
                     (let [l-nh (or (:num-heads layer-w) num-heads)
                           l-nkv (or (:num-kv-heads layer-w) num-kv-heads)
                           [h-next new-kv] (gemma-block h layer-w l-nh l-nkv pos-ids layer-kv pos)]
                       [h-next (conj updated-acc new-kv)]))
                   [tok-embed []]
                   (map vector layers-with-ple kv-caches))
           [(reduce (fn [h layer-w]
                      (let [l-nh (or (:num-heads layer-w) num-heads)
                            l-nkv (or (:num-kv-heads layer-w) num-kv-heads)]
                        (gemma-block h layer-w l-nh l-nkv pos-ids)))
                    tok-embed
                    layers-with-ple)
            nil])
         normed (rms-norm x-out final-norm-w 1e-6)
         embed-t (transpose embed-tokens [1 0])
         raw-logits (linear normed embed-t nil)
         final-softcap (get opts :final-logit-softcap 30.0)
         capped-logits (if (and (number? final-softcap) (pos? final-softcap))
                         (* final-softcap (tanh (/ raw-logits final-softcap)))
                         raw-logits)]
     (if use-kv?
       [capped-logits new-kv-caches]
       capped-logits))))

(defn weight-key-map
  "Returns Gemma 1 / Gemma 2 safetensors weight key mapping for layer `layer-idx`."
  [layer-idx]
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

(defn gemma3-weight-key-map
  "Returns Gemma 3 safetensors weight key mapping for layer `layer-idx` including QK norm parameters."
  [layer-idx]
  (let [prefix (str "model.layers." layer-idx ".")]
    {:input-ln-w     (str prefix "input_layernorm.weight")
     :q-w            (str prefix "self_attn.q_proj.weight")
     :k-w            (str prefix "self_attn.k_proj.weight")
     :v-w            (str prefix "self_attn.v_proj.weight")
     :o-w            (str prefix "self_attn.o_proj.weight")
     :q-norm-w       (str prefix "self_attn.q_norm.weight")
     :k-norm-w       (str prefix "self_attn.k_norm.weight")
     :post-attn-ln-w (str prefix "post_attention_layernorm.weight")
     :pre-mlp-ln-w   (str prefix "pre_feedforward_layernorm.weight")
     :post-mlp-ln-w  (str prefix "post_feedforward_layernorm.weight")
     :gate-w         (str prefix "mlp.gate_proj.weight")
     :up-w           (str prefix "mlp.up_proj.weight")
     :down-w         (str prefix "mlp.down_proj.weight")}))

(defn gemma4-weight-key-map
  "Returns Gemma 4 safetensors weight key mapping for layer `layer-idx` including per-layer input keys."
  ([layer-idx] (gemma4-weight-key-map layer-idx "model.layers."))
  ([layer-idx prefix-base]
   (let [prefix (if (.endsWith ^String prefix-base ".")
                  (str prefix-base layer-idx ".")
                  (str prefix-base "." layer-idx "."))]
     {:input-ln-w             (str prefix "input_layernorm.weight")
      :layer-scalar-w         (str prefix "layer_scalar")
      :q-w                    (str prefix "self_attn.q_proj.weight")
      :k-w                    (str prefix "self_attn.k_proj.weight")
      :v-w                    (str prefix "self_attn.v_proj.weight")
      :o-w                    (str prefix "self_attn.o_proj.weight")
      :q-norm-w               (str prefix "self_attn.q_norm.weight")
      :k-norm-w               (str prefix "self_attn.k_norm.weight")
      :post-attn-ln-w         (str prefix "post_attention_layernorm.weight")
      :pre-mlp-ln-w           (str prefix "pre_feedforward_layernorm.weight")
      :post-mlp-ln-w          (str prefix "post_feedforward_layernorm.weight")
      :gate-w                 (str prefix "mlp.gate_proj.weight")
      :up-w                   (str prefix "mlp.up_proj.weight")
      :down-w                 (str prefix "mlp.down_proj.weight")
      :per-layer-gate-w       (str prefix "per_layer_input_gate.weight")
      :per-layer-proj-w       (str prefix "per_layer_projection.weight")
      :post-per-layer-norm-w  (str prefix "post_per_layer_input_norm.weight")})))

