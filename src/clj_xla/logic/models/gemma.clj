(ns clj-xla.logic.models.gemma
  "Declarative Gemma 4 Architecture definition in pure Tensor Logic Hiccup AST."
  (:require [clj-xla.models.gemma :as gemma-legacy]))

(defn gemma4-layer-ast
  "Generates Tensor Logic Hiccup AST for Gemma 4 Transformer layer block `layer-idx`."
  ([layer-idx max-seq-len config]
   (gemma4-layer-ast layer-idx max-seq-len config {}))
  ([layer-idx max-seq-len config layer-opts]
   (let [i layer-idx
         {:keys [num-heads num-kv-heads pl-dim total-pl-dim layer-types]} config
         is-shared? (:is-shared? layer-opts)
         shared-k (:shared-k layer-opts)
         shared-v (:shared-v layer-opts)
         num-heads (long (or num-heads 8))
         num-kv-heads (long (or num-kv-heads 1))
         pl-dim (long (or pl-dim 256))
         total-pl-dim (long (or total-pl-dim (* 35 pl-dim)))
         has-ple? (pos? total-pl-dim)

         l-type (if layer-types (nth layer-types i nil) nil)
         is-global? (or (= l-type "full_attention")
                        (= l-type :full_attention)
                        (zero? (mod (inc i) 5)))
         head-dim (long (if is-global?
                          (or (:global-head-dim config) 512)
                          (or (:head-dim config) 256)))
         q-dim (* num-heads head-dim)
         kv-dim (* num-kv-heads head-dim)
         group-size (quot num-heads num-kv-heads)
         rope-prop (if is-global? 0.25 1.0)
         theta (if is-global? 1000000.0 10000.0)
         window (if is-global? nil 512)

         h-in (keyword (str "h" i))
         h-out (keyword (str "h" (inc i)))

         input-ln-w (keyword (str "input_ln_w_" i))
         layer-scalar-w (keyword (str "layer_scalar_" i))
         q-w (keyword (str "q_w_" i))
         k-w (keyword (str "k_w_" i))
         v-w (keyword (str "v_w_" i))
         o-w (keyword (str "o_w_" i))
         q-norm-w (keyword (str "q_norm_w_" i))
         k-norm-w (keyword (str "k_norm_w_" i))
         post-attn-ln-w (keyword (str "post_attn_ln_w_" i))
         pre-mlp-ln-w (keyword (str "pre_mlp_ln_w_" i))
         post-mlp-ln-w (keyword (str "post_mlp_ln_w_" i))
         gate-w (keyword (str "gate_w_" i))
         up-w (keyword (str "up_w_" i))
         down-w (keyword (str "down_w_" i))
         per-layer-gate-w (keyword (str "per_layer_gate_w_" i))
         per-layer-proj-w (keyword (str "per_layer_proj_w_" i))
         post-per-layer-norm-w (keyword (str "post_per_layer_norm_w_" i))
         pl-in-var (keyword (str "pl_in_" i))

         x-norm1 (keyword (str "x_norm1_" i))
         q-raw (keyword (str "q_raw_" i))
         k-raw (keyword (str "k_raw_" i))
         v-raw (keyword (str "v_raw_" i))
         v-normed (keyword (str "v_normed_" i))
         q-heads-raw (keyword (str "q_heads_raw_" i))
         k-heads-raw (keyword (str "k_heads_raw_" i))
         v-heads (keyword (str "v_heads_" i))
         q-normed-4d (keyword (str "q_normed_4d_" i))
         k-normed-4d (keyword (str "k_normed_4d_" i))
         q-normed-3d (keyword (str "q_normed_3d_" i))
         k-normed-3d (keyword (str "k_normed_3d_" i))
         q-rope (keyword (str "q_rope_" i))
         k-rope (keyword (str "k_rope_" i))
         q-ro (keyword (str "q_ro_" i))
         k-ro (keyword (str "k_ro_" i))
         actual-k-ro (if is-shared? shared-k k-ro)
         actual-v-heads (if is-shared? shared-v v-heads)
         k-rep (keyword (str "k_rep_" i))
         k-heads (keyword (str "k_heads_" i))
         v-rep (keyword (str "v_rep_" i))
         v-rep-heads (keyword (str "v_rep_heads_" i))
         scores (keyword (str "scores_" i))
         probs (keyword (str "probs_" i))
         ctx (keyword (str "ctx_" i))
         ctx-flat (keyword (str "ctx_flat_" i))
         attn-raw (keyword (str "attn_raw_" i))
         attn-normed (keyword (str "attn_normed_" i))
         res1 (keyword (str "res1_" i))

         x-norm2 (keyword (str "x_norm2_" i))
         gate (keyword (str "gate_" i))
         up (keyword (str "up_" i))
         mlp-act (keyword (str "mlp_act_" i))
         mlp-raw (keyword (str "mlp_raw_" i))
         mlp-normed (keyword (str "mlp_normed_" i))
         res2 (keyword (str "res2_" i))

         gate-raw (keyword (str "gate_raw_" i))
         gated (keyword (str "gated_" i))
         proj-raw (keyword (str "proj_raw_" i))
         ple-normed (keyword (str "ple_normed_" i))
         res3 (keyword (str "res3_" i))]

     [:block {:name (keyword (str "gemma4_layer_" i))}
      ;; 1. Pre-Attention RMSNorm
      [:rms-norm [x-norm1 :b :p :d] [h-in :b :p :d] [input-ln-w :d] {:eps 1e-6}]

      ;; 2. Q Projection & Reshape
      [:= [q-raw :b :p :qd] [x-norm1 :b :p :d] [q-w :qd :d]]
      [:reshape [q-heads-raw :b :p :h :dh] [q-raw :b :p :qd] {:shape [1 max-seq-len num-heads head-dim]}]
      [:rms-norm [q-normed-4d :b :p :h :dh] [q-heads-raw :b :p :h :dh] [q-norm-w :dh] {:eps 1e-6}]
      [:reshape [q-normed-3d :b :p :qd] [q-normed-4d :b :p :h :dh] {:shape [1 max-seq-len q-dim]}]
      [:rope [q-rope :b :p :qd] [q-normed-3d :b :p :qd] {:head-dim head-dim :theta theta :rope-proportion rope-prop}]
      [:reshape [q-ro :b :p :h :dh] [q-rope :b :p :qd] {:shape [1 max-seq-len num-heads head-dim]}]

      ;; 3. K, V Projections (computed only if not shared)
      (when-not is-shared?
        [:block {:name (keyword (str "kv_proj_" i))}
         [:= [k-raw :b :p :kvd] [x-norm1 :b :p :d] [k-w :kvd :d]]
         [:= [v-raw :b :p :kvd] [x-norm1 :b :p :d] [v-w :kvd :d]]
         [:rms-norm [v-normed :b :p :kvd] [v-raw :b :p :kvd] {:eps 1e-6}]
         [:reshape [k-heads-raw :b :p :kvh :dh] [k-raw :b :p :kvd] {:shape [1 max-seq-len num-kv-heads head-dim]}]
         [:reshape [v-heads :b :p :kvh :dh] [v-normed :b :p :kvd] {:shape [1 max-seq-len num-kv-heads head-dim]}]
         [:rms-norm [k-normed-4d :b :p :kvh :dh] [k-heads-raw :b :p :kvh :dh] [k-norm-w :dh] {:eps 1e-6}]
         [:reshape [k-normed-3d :b :p :kvd] [k-normed-4d :b :p :kvh :dh] {:shape [1 max-seq-len kv-dim]}]
         [:rope [k-rope :b :p :kvd] [k-normed-3d :b :p :kvd] {:head-dim head-dim :theta theta :rope-proportion rope-prop}]
         [:reshape [k-ro :b :p :kvh :dh] [k-rope :b :p :kvd] {:shape [1 max-seq-len num-kv-heads head-dim]}]])

      ;; 4. Broadcast KV Heads
      [:= [k-rep :b :p :kvh :g :dh] [actual-k-ro :b :p :kvh :dh] {:shape [1 max-seq-len num-kv-heads group-size head-dim]}]
      [:reshape [k-heads :b :p :h :dh] [k-rep :b :p :kvh :g :dh] {:shape [1 max-seq-len num-heads head-dim]}]
      [:= [v-rep :b :p :kvh :g :dh] [actual-v-heads :b :p :kvh :dh] {:shape [1 max-seq-len num-kv-heads group-size head-dim]}]
      [:reshape [v-rep-heads :b :p :h :dh] [v-rep :b :p :kvh :g :dh] {:shape [1 max-seq-len num-heads head-dim]}]

      ;; 5. Scaled Dot-Product Attention: QK^T -> Softmax (sliding or full) -> probs @ V
      [:= [scores :b :h :p-q :p-k] {:scale 1.0} [q-ro :b :p-q :h :dh] [k-heads :b :p-k :h :dh]]
      [:causal-softmax [probs :b :h :p-q :p-k] [scores :b :h :p-q :p-k] (if window {:sliding-window window} {})]
      [:= [ctx :b :p-q :h :dh] [probs :b :h :p-q :p-k] [v-rep-heads :b :p-k :h :dh]]
      [:reshape [ctx-flat :b :p :qd] [ctx :b :p-q :h :dh] {:shape [1 max-seq-len q-dim]}]

      ;; 6. Output Projection & Post-Attention RMSNorm
      [:= [attn-raw :b :p :d] [ctx-flat :b :p :qd] [o-w :d :qd]]
      [:rms-norm [attn-normed :b :p :d] [attn-raw :b :p :d] [post-attn-ln-w :d] {:eps 1e-6}]

      ;; 7. Residual Connection 1
      [:= [res1 :b :p :d] [h-in :b :p :d]]
      [:= [res1 :b :p :d] [attn-normed :b :p :d]]

      ;; 8. Pre-MLP RMSNorm
      [:rms-norm [x-norm2 :b :p :d] [res1 :b :p :d] [pre-mlp-ln-w :d] {:eps 1e-6}]

      ;; 9. GeGLU MLP Block: down_proj(gelu(gate_proj(x)) * up_proj(x))
      [:= [gate :b :p :dff] {:act :gelu} [x-norm2 :b :p :d] [gate-w :dff :d]]
      [:= [up :b :p :dff] [x-norm2 :b :p :d] [up-w :dff :d]]
      [:= [mlp-act :b :p :dff] [gate :b :p :dff] [up :b :p :dff]]
      [:= [mlp-raw :b :p :d] [mlp-act :b :p :dff] [down-w :d :dff]]
      [:rms-norm [mlp-normed :b :p :d] [mlp-raw :b :p :d] [post-mlp-ln-w :d] {:eps 1e-6}]

      ;; 10. Residual Connection 2
      [:= [res2 :b :p :d] [res1 :b :p :d]]
      [:= [res2 :b :p :d] [mlp-normed :b :p :d]]

      ;; 11. Gemma 4 Per-Layer Input (PLE) Gating Sub-block (if has-ple?)
      (when has-ple?
        [:block {:name (keyword (str "ple_gate_" i))}
         [:= [gate-raw :b :p :pld] {:act :sigmoid} [res2 :b :p :d] [per-layer-gate-w :pld :d]]
         [:= [gated :b :p :pld] [gate-raw :b :p :pld] [pl-in-var :b :p :pld]]
         [:= [proj-raw :b :p :d] [gated :b :p :pld] [per-layer-proj-w :d :pld]]
         [:rms-norm [ple-normed :b :p :d] [proj-raw :b :p :d] [post-per-layer-norm-w :d] {:eps 1e-6}]
         [:= [res3 :b :p :d] [res2 :b :p :d]]
         [:= [res3 :b :p :d] [ple-normed :b :p :d]]])

      ;; 12. Gemma 4 Layer Scalar
      [:= [h-out :b :p :d] [(if has-ple? res3 res2) :b :p :d] [layer-scalar-w :one]]])))

(defn gemma4-model-ast
  "Generates full Gemma 4 model forward pass in pure Tensor Logic Hiccup AST."
  [config]
  (let [cfg (merge (gemma-legacy/gemma4-config :e2b) config)
        {:keys [num-layers max-seq-len hidden-dim pl-dim total-pl-dim final-logit-softcap num-kv-shared-layers]} cfg
        num-layers (long (or num-layers 35))
        num-kv-shared (long (or num-kv-shared-layers 0))
        num-unshared (- num-layers num-kv-shared)
        has-shared-kv? (and (pos? num-unshared) (pos? num-kv-shared))
        max-seq-len (long (or max-seq-len 128))
        hidden-dim (long (or hidden-dim 1536))
        pl-dim (long (or pl-dim 256))
        total-pl-dim (long (or total-pl-dim (* num-layers pl-dim)))
        has-ple? (pos? total-pl-dim)
        h-final (keyword (str "h" num-layers))

        last-unshared-sliding (when has-shared-kv? (last (filter #(not= (mod (inc %) 5) 0) (range num-unshared))))
        last-unshared-full (when has-shared-kv? (last (filter #(= (mod (inc %) 5) 0) (range num-unshared))))]
    [:block {:name :full_gemma4_model}
     ;; 1. Token Embedding Lookup (scaled by sqrt(hidden-dim))
     [:gather [:tok_embed_raw :b :p :d] [:embed_tokens :v :d] [:x :b :p]]
     [:= [:h0 :b :p :d] {:scale (Math/sqrt (double hidden-dim))} [:tok_embed_raw :b :p :d]]

     ;; 2. Gemma 4 Per-Layer Embedding (PLE) generation
     (when has-ple?
       [:block {:name :ple_generation}
        [:gather [:raw_pl_tok :b :p :total_pl_dim] [:embed_tokens_per_layer :v :total_pl_dim] [:x :b :p]]
        [:= [:pl_tok_scaled :b :p :total_pl_dim] {:scale 16.0} [:raw_pl_tok :b :p :total_pl_dim]]
        [:= [:pl_context_raw :b :p :total_pl_dim] [:h0 :b :p :d] [:per_layer_model_projection :total_pl_dim :d]]
        [:reshape [:pl_tok_4d :b :p :l :pld] [:pl_tok_scaled :b :p :total_pl_dim] {:shape [1 max-seq-len num-layers pl-dim]}]
        [:reshape [:pl_context_4d :b :p :l :pld] [:pl_context_raw :b :p :total_pl_dim] {:shape [1 max-seq-len num-layers pl-dim]}]
        [:rms-norm [:pl_context_norm :b :p :l :pld] [:pl_context_4d :b :p :l :pld] [:per_layer_projection_norm :pld] {:eps 1e-6}]
        [:= [:pl_sum :b :p :l :pld] [:pl_context_norm :b :p :l :pld]]
        [:= [:pl_sum :b :p :l :pld] [:pl_tok_4d :b :p :l :pld]]
        [:= [:ple_all :b :p :l :pld] {:scale (/ 1.0 (Math/sqrt 2.0))} [:pl_sum :b :p :l :pld]]
        ;; Layer PLE Slices
        (mapv (fn [i]
                (let [sl-var (keyword (str "pl_slice_" i))
                      in-var (keyword (str "pl_in_" i))]
                  [:block {:name (keyword (str "ple_slice_block_" i))}
                   [:slice [sl-var :b :p :one :pld] [:ple_all :b :p :l :pld] {:start [0 0 i 0] :limit [1 max-seq-len (inc i) pl-dim]}]
                   [:reshape [in-var :b :p :pld] [sl-var :b :p :one :pld] {:shape [1 max-seq-len pl-dim]}]]))
              (range num-layers))])

     ;; 3. Sequential Transformer Layer Blocks
     (mapv (fn [i]
             (let [is-shared? (and has-shared-kv? (>= i num-unshared))
                   is-global? (zero? (mod (inc i) 5))
                   shared-k (when is-shared?
                              (if is-global?
                                (keyword (str "k_ro_" last-unshared-full))
                                (keyword (str "k_ro_" last-unshared-sliding))))
                   shared-v (when is-shared?
                              (if is-global?
                                (keyword (str "v_heads_" last-unshared-full))
                                (keyword (str "v_heads_" last-unshared-sliding))))]
               (gemma4-layer-ast i max-seq-len cfg {:is-shared? is-shared?
                                                    :shared-k shared-k
                                                    :shared-v shared-v})))
           (range num-layers))

     ;; 4. Final RMSNorm
     [:rms-norm [:normed :b :p :d] [h-final :b :p :d] [:final_norm_w :d] {:eps 1e-6}]

     ;; 5. Tied LM Head with optional final logit softcapping (30.0 * tanh(x / 30.0))
     [:= [:logits :b :p :v] (if (and (number? final-logit-softcap) (pos? final-logit-softcap))
                              {:softcap (double final-logit-softcap)}
                              {})
      [:normed :b :p :d] [:embed_tokens :v :d]]]))
