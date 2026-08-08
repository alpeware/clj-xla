(ns clj-xla.models.gemma
  "Gemma 1, Gemma 2, Gemma 3, and Gemma 4 model architecture implementations in pure Clojure/StableHLO."
  (:refer-clojure :exclude [+ * /])
  (:require [clj-xla.nn.activations :refer [gelu]]
            [clj-xla.nn.attention :refer [apply-rope gqa-causal-attention]]
            [clj-xla.nn.norm :refer [gemma-rms-norm rms-norm]]
            [clj-xla.tensor :refer [* + / dot-general emit-constant! gather matmul reshape slice tanh transpose]]))
(defn linear [x w b]
  (let [tx (emit-constant! x nil)
        tw (emit-constant! w nil)
        [_ x-shape _] (:type tx)
        [_ w-shape _] (:type tw)
        x-rank (count x-shape)
        w-rank (count w-shape)
        x-in-dim (last x-shape)
        w-dim0 (first w-shape)
        out (if (and (= w-rank 2) (= w-dim0 x-in-dim))
              (matmul tx tw)
              (dot-general tx tw {:contracting_dims {:lhs [(dec x-rank)] :rhs [(dec w-rank)]}}))]
    (if (some? b)
      (+ out b)
      out)))

(def DEFAULT_GEMMA_CONFIG
  {:hidden-dim 2048
   :intermediate-dim 16384
   :num-layers 18
   :num-heads 8
   :num-kv-heads 1
   :head-dim 256
   :vocab-size 256000
   :norm-eps 1e-6})

(def DEFAULT_GEMMA3_270M_CONFIG
  {:hidden-dim 640
   :intermediate-dim 2048
   :num-layers 18
   :num-heads 4
   :num-kv-heads 1
   :head-dim 160
   :vocab-size 262144
   :norm-eps 1e-6})

(def DEFAULT_GEMMA4_E2B_CONFIG
  {:hidden-dim 1536
   :intermediate-dim 6144
   :pl-dim 256
   :total-pl-dim 8960 ;; 35 * 256
   :num-layers 35
   :num-heads 8
   :num-kv-heads 1
   :head-dim 256
   :vocab-size 262144
   :norm-eps 1e-6
   :final-logit-softcap 30.0
   :num-kv-shared-layers 20
   :layer-types ["sliding_attention" "sliding_attention" "sliding_attention" "sliding_attention" "full_attention"
                 "sliding_attention" "sliding_attention" "sliding_attention" "sliding_attention" "full_attention"
                 "sliding_attention" "sliding_attention" "sliding_attention" "sliding_attention" "full_attention"
                 "sliding_attention" "sliding_attention" "sliding_attention" "sliding_attention" "full_attention"
                 "sliding_attention" "sliding_attention" "sliding_attention" "sliding_attention" "full_attention"
                 "sliding_attention" "sliding_attention" "sliding_attention" "sliding_attention" "full_attention"
                 "sliding_attention" "sliding_attention" "sliding_attention" "sliding_attention" "full_attention"]})

(def DEFAULT_GEMMA4_E4B_CONFIG
  {:hidden-dim 2560
   :intermediate-dim 10240
   :pl-dim 256
   :total-pl-dim 10752 ;; 42 * 256
   :num-layers 42
   :num-heads 8
   :num-kv-heads 2
   :head-dim 256
   :global-head-dim 512
   :vocab-size 262144
   :norm-eps 1e-6
   :final-logit-softcap 30.0
   :num-kv-shared-layers 18})

(def DEFAULT_GEMMA4_12B_CONFIG
  {:hidden-dim 3840
   :intermediate-dim 15360
   :pl-dim 0
   :total-pl-dim 0
   :num-layers 48
   :num-heads 16
   :num-kv-heads 8
   :num-global-kv-heads 1
   :head-dim 256
   :global-head-dim 512
   :vocab-size 262144
   :norm-eps 1e-6
   :final-logit-softcap 30.0
   :num-kv-shared-layers 0})

(defn gemma-config
  "Returns Gemma 1/2 configuration map with optional custom overrides."
  ([] DEFAULT_GEMMA_CONFIG)
  ([overrides] (merge DEFAULT_GEMMA_CONFIG overrides)))

(defn gemma3-config
  "Returns Gemma 3 configuration map with optional custom overrides."
  ([] DEFAULT_GEMMA3_270M_CONFIG)
  ([overrides] (merge DEFAULT_GEMMA3_270M_CONFIG overrides)))

(defn gemma4-config
  "Returns Gemma 4 configuration map for the specified variant (e.g. :e2b, :e4b, :12b) with optional custom overrides."
  ([] (gemma4-config :e2b {}))
  ([variant-or-overrides]
   (if (keyword? variant-or-overrides)
     (gemma4-config variant-or-overrides {})
     (gemma4-config :e2b variant-or-overrides)))
  ([variant overrides]
   (let [base (case variant
                :12b DEFAULT_GEMMA4_12B_CONFIG
                :e4b DEFAULT_GEMMA4_E4B_CONFIG
                :e2b DEFAULT_GEMMA4_E2B_CONFIG
                DEFAULT_GEMMA4_E2B_CONFIG)]
     (merge base overrides))))

(defn embed-lookup [x embed-tokens hidden-dim]
  (let [raw-embed (gather embed-tokens x)
        h-dim (or hidden-dim (second (:type embed-tokens)) 1536)
        scale (Math/sqrt (double h-dim))]
    (* raw-embed scale)))

(defn gemma-mlp
  "Gemma GeGLU MLP Block: down_proj(gelu(gate_proj(x)) * up_proj(x))."
  [x gate-w up-w down-w]
  (let [gate-t (transpose gate-w [1 0])
        up-t (transpose up-w [1 0])
        down-t (transpose down-w [1 0])
        gate-out (gelu (linear x gate-t nil))
        up-out (linear x up-t nil)
        hidden (* gate-out up-out)]
    (linear hidden down-t nil)))

(defn gemma-attention
  "Multi-head / Grouped-Query Causal Self-Attention block for Gemma 1/2/3/4."
  ([x q-w k-w v-w o-w num-heads _num-kv-heads pos-ids]
   (gemma-attention x q-w k-w v-w o-w num-heads _num-kv-heads pos-ids nil nil {}))
  ([x q-w k-w v-w o-w num-heads _num-kv-heads pos-ids past-kv pos]
   (gemma-attention x q-w k-w v-w o-w num-heads _num-kv-heads pos-ids past-kv pos {}))
  ([x q-w k-w v-w o-w num-heads _num-kv-heads pos-ids past-kv pos opts]
   (let [{:keys [q-norm-w k-norm-w theta-base attn-softcap rotary-dim rope-proportion shared-kv]} opts
         norm-fn (or (:norm-fn opts) (if (some? q-norm-w) rms-norm gemma-rms-norm))
         theta-base (or theta-base 10000.0)
         rope-proportion (or rope-proportion 1.0)
         q (linear x (transpose q-w [1 0]) nil)
         [_ [batch seq-len q-proj-dim] _] (:type q)
         h-dim (or (:head-dim opts) (quot q-proj-dim num-heads))
         q-4d (transpose (reshape q [batch seq-len num-heads h-dim]) [0 2 1 3])
         q-normed (if (some? q-norm-w) (norm-fn q-4d q-norm-w 1e-6) q-4d)

         ;; Compute or reuse Shared KV
         [k-rope-3d v computed-kv]
         (if (some? shared-kv)
           [(first shared-kv) (second shared-kv) shared-kv]
           (let [k-w-t (transpose k-w [1 0])
                 v-w-t (transpose v-w [1 0])
                 k (linear x k-w-t nil)
                 v-raw (linear x v-w-t nil)
                 v (if (some? q-norm-w) (rms-norm v-raw nil 1e-6) v-raw)
                 [_ [_ _ kv-dim] _] (:type k)
                 l-nkv (quot kv-dim h-dim)
                 k-4d (transpose (reshape k [batch seq-len l-nkv h-dim]) [0 2 1 3])
                 k-normed (if (some? k-norm-w) (norm-fn k-4d k-norm-w 1e-6) k-4d)
                 [_ k-rope] (apply-rope q-normed k-normed pos-ids h-dim theta-base rotary-dim nil rope-proportion)
                 k-3d (reshape (transpose k-rope [0 2 1 3]) [batch seq-len kv-dim])]
             [k-3d v [k-3d v]]))

         [q-rope _] (apply-rope q-normed q-normed pos-ids h-dim theta-base rotary-dim nil rope-proportion)
         q-rope-3d (reshape (transpose q-rope [0 2 1 3]) [batch seq-len q-proj-dim])
         attn-scale (get opts :scale (if (some? q-norm-w) 1.0 (/ 1.0 (Math/sqrt (double h-dim)))))
         attn-opts {:scale attn-scale}
         [_ [_ _ kv-dim] _] (:type k-rope-3d)
         l-nkv (quot kv-dim h-dim)
         attn-res (if (some? past-kv)
                    (gqa-causal-attention q-rope-3d k-rope-3d v o-w num-heads l-nkv attn-softcap past-kv pos attn-opts)
                    (gqa-causal-attention q-rope-3d k-rope-3d v o-w num-heads l-nkv attn-softcap nil nil attn-opts))]
     (if (vector? attn-res)
       [(first attn-res) (second attn-res) computed-kv]
       [attn-res nil computed-kv]))))

(defn gemma-block
  "Single Gemma 1/2/3/4 Transformer block with RMSNorms, optional QK Norm, and dual residual connections."
  ([x weights num-heads num-kv-heads pos-ids]
   (gemma-block x weights num-heads num-kv-heads pos-ids nil nil))
  ([x weights num-heads num-kv-heads pos-ids past-kv pos]
   (let [{:keys [input-ln-w q-w k-w v-w o-w q-norm-w k-norm-w
                 post-attn-ln-w pre-mlp-ln-w post-mlp-ln-w
                 gate-w up-w down-w theta-base attn-softcap rotary-dim rope-proportion
                 per-layer-gate-w per-layer-proj-w post-per-layer-norm-w
                 per-layer-input layer-scalar-w shared-kv]
          :or {theta-base 10000.0 attn-softcap nil rope-proportion 1.0}} weights
         norm-fn (or (:norm-fn weights) (if (some? q-norm-w) rms-norm gemma-rms-norm))
         ;; 1. Attention Sub-block
         x-norm1 (norm-fn x input-ln-w 1e-6)
         attn-opts {:q-norm-w q-norm-w
                    :k-norm-w k-norm-w
                    :theta-base theta-base
                    :attn-softcap attn-softcap
                    :rotary-dim rotary-dim
                    :full-dim (:full-dim weights)
                    :head-dim (:head-dim weights)
                    :rope-proportion rope-proportion
                    :norm-fn norm-fn
                    :shared-kv shared-kv}
         [attn-raw updated-kv computed-kv] (gemma-attention x-norm1 q-w k-w v-w o-w num-heads num-kv-heads pos-ids past-kv pos attn-opts)
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
                         act-gate (gelu gate-raw)
                         gated (* act-gate per-layer-input)
                         proj-raw (linear gated pl-proj nil)
                         normed (if post-per-layer-norm-w (norm-fn proj-raw post-per-layer-norm-w 1e-6) proj-raw)]
                     (+ mlp-out normed))
                   mlp-out)

         ;; 4. Gemma 4 Layer Scalar (skip scale)
         x-out (if (some? layer-scalar-w)
                 (* ple-out layer-scalar-w)
                 ple-out)]
     (cond
       (some? past-kv) [x-out updated-kv computed-kv]
       (:return-shared-kv? weights) [x-out nil computed-kv]
       :else x-out))))

(defn full-gemma-forward
  "Full Gemma 1/2/3 Transformer forward pass (embedding lookup + layers + final norm + lm_head)."
  ([x embed-tokens layers-weights final-norm-w pos-ids]
   (full-gemma-forward x embed-tokens layers-weights final-norm-w pos-ids 8 1 nil nil {}))
  ([x embed-tokens layers-weights final-norm-w pos-ids num-heads num-kv-heads]
   (full-gemma-forward x embed-tokens layers-weights final-norm-w pos-ids num-heads num-kv-heads nil nil {}))
  ([x embed-tokens layers-weights final-norm-w pos-ids num-heads num-kv-heads kv-caches pos]
   (full-gemma-forward x embed-tokens layers-weights final-norm-w pos-ids num-heads num-kv-heads kv-caches pos {}))
  ([x embed-tokens layers-weights final-norm-w pos-ids num-heads num-kv-heads kv-caches pos opts]
   (let [[_ [_seq-len] _] (:type x)
         [_ [_vocab-size hidden-dim] _] (:type embed-tokens)
         tok-embed (embed-lookup x embed-tokens hidden-dim)
         use-kv? (some? kv-caches)
         [x-out new-kv-caches]
         (if use-kv?
           (reduce (fn [[h updated-acc] [layer-w layer-kv]]
                     (let [[h-next new-kv _] (gemma-block h layer-w num-heads num-kv-heads pos-ids layer-kv pos)]
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
  "Full Gemma 4 Transformer forward pass including Per-Layer Embeddings (PLE), KV Sharing, and Layer Scalars."
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
         num-kv-shared (or (:num-kv-shared-layers opts) 0)
         num-unshared (- num-layers num-kv-shared)
         tok-embed (embed-lookup x embed-tokens hidden-dim)
         ;; Compute PLE (Per-Layer Embedding) representation if pl-dim > 0
         pl-shape (when (some? embed-per-layer) (second (:type embed-per-layer)))
         total-pl-dim (if (seq pl-shape) (second pl-shape) 0)
         pl-dim (if (and (pos? total-pl-dim) (pos? num-layers)) (quot total-pl-dim num-layers) 0)
         has-ple? (pos? pl-dim)

         ple-all (when has-ple?
                   (let [raw-pl-tok (gather embed-per-layer x)
                         pl-tok-scaled (* raw-pl-tok 16.0)
                         pl-proj-t (transpose per-layer-model-proj-w [1 0])
                         pl-context-raw (linear tok-embed pl-proj-t nil)
                         pl-tok-4d (reshape pl-tok-scaled [batch seq-len num-layers pl-dim])
                         pl-context-4d (reshape pl-context-raw [batch seq-len num-layers pl-dim])
                         pl-context-norm (rms-norm pl-context-4d per-layer-proj-norm-w 1e-6)]
                     (* (+ pl-context-norm pl-tok-4d) (/ 1.0 (Math/sqrt 2.0)))))

         layers-with-ple (mapv (fn [i lw]
                                 (let [pl-input (when has-ple?
                                                  (let [pl-slice (slice ple-all [0 0 i 0] [batch seq-len (inc i) pl-dim] [1 1 1 1])]
                                                    (reshape pl-slice [batch seq-len pl-dim])))
                                       l-type (or (:layer-type lw)
                                                  (if (:is-global? lw) :full_attention nil)
                                                  (if (zero? (mod (inc i) 5)) :full_attention :sliding_attention))]
                                   (assoc lw :per-layer-input pl-input :norm-fn rms-norm :layer-type l-type :layer-idx i)))
                               (range num-layers)
                               layers-weights)
         use-kv? (some? kv-caches)

         ;; Sequential layer pass with KV sharing state tracking
         [x-out new-kv-caches _]
         (reduce (fn [[h updated-kv-acc shared-kv-store] [i layer-w layer-kv]]
                   (let [l-nh (or (:num-heads layer-w) num-heads)
                         l-nkv (or (:num-kv-heads layer-w) num-kv-heads)
                         l-type (:layer-type layer-w)
                         is-shared? (if (contains? layer-w :is-shared?)
                                      (:is-shared? layer-w)
                                      (>= i num-unshared))
                         kv-to-pass (if is-shared? (get shared-kv-store l-type) nil)
                         w-with-kv (assoc layer-w :shared-kv kv-to-pass :return-shared-kv? true)
                         [h-next new-kv computed-kv] (if use-kv?
                                                       (gemma-block h w-with-kv l-nh l-nkv pos-ids layer-kv pos)
                                                       (gemma-block h w-with-kv l-nh l-nkv pos-ids))
                         next-shared-store (if (and (not is-shared?) (some? computed-kv))
                                             (assoc shared-kv-store l-type computed-kv)
                                             shared-kv-store)
                         next-kv-acc (if use-kv? (conj updated-kv-acc new-kv) updated-kv-acc)]
                     [h-next next-kv-acc next-shared-store]))
                 [tok-embed [] {}]
                 (map vector (range num-layers) layers-with-ple (or kv-caches (repeat num-layers nil))))

         normed (rms-norm x-out final-norm-w 1e-6)
         normed-last (if-let [idx (:last-token-idx opts)]
                       (let [[_ [_ _ h-dim] _] (:type normed)]
                         (reshape (slice normed [0 idx 0] [1 (inc idx) h-dim] [1 1 1]) [1 1 h-dim]))
                       normed)
         embed-t (transpose embed-tokens [1 0])
         raw-logits (linear normed-last embed-t nil)
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
