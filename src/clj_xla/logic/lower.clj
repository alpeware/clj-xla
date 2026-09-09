(ns clj-xla.logic.lower
  "Lowering engine from normalized Tensor Logic AST to StableHLO EDN SSA graph."
  (:require [clj-xla.logic.ast :as ast]
            [clj-xla.logic.dce :as dce]
            [clj-xla.logic.expand :as expand]
            [clj-xla.logic.index :as idx]
            [clj-xla.logic.shape :as shape]
            [clj-xla.stablehlo :as shlo]))

(defn- gen-id [prefix counter]
  (keyword (str prefix "_" (swap! counter inc))))

(defn- indices->dim-numbers [ordered-idxs target-idxs]
  (let [idx->pos (into {} (map-indexed (fn [pos k] [k pos]) ordered-idxs))]
    (mapv #(get idx->pos %) target-idxs)))

(defn- emit-post-activation! [eqns-atom counter in-var out-var attrs dtype]
  (let [scale (:scale attrs)
        act (:act attrs)
        softcap (:softcap attrs)]
    (cond
      scale
      (let [c-var (gen-id "c_scale" counter)
            c-eqn {:op :stablehlo/constant :value (double scale) :outvars [c-var]}
            mul-var (if (or act softcap) (gen-id "t_scaled" counter) out-var)
            mul-eqn {:op :stablehlo/multiply :invars [in-var c-var] :outvars [mul-var]}]
        (swap! eqns-atom conj c-eqn mul-eqn)
        (if (or act softcap)
          (emit-post-activation! eqns-atom counter mul-var out-var (dissoc attrs :scale) dtype)
          mul-var))

      (or (= act :sigmoid) (= act :logistic))
      (let [log-eqn {:op :stablehlo/logistic :invars [in-var] :outvars [out-var]}]
        (swap! eqns-atom conj log-eqn)
        out-var)

      (or (= act :silu) (= act :swish))
      (let [log-var (gen-id "t_logistic" counter)
            log-eqn {:op :stablehlo/logistic :invars [in-var] :outvars [log-var]}
            mul-eqn {:op :stablehlo/multiply :invars [in-var log-var] :outvars [out-var]}]
        (swap! eqns-atom conj log-eqn mul-eqn)
        out-var)

      (= act :tanh)
      (let [t-eqn {:op :stablehlo/tanh :invars [in-var] :outvars [out-var]}]
        (swap! eqns-atom conj t-eqn)
        out-var)

      (= act :gelu)
      (let [c-half (gen-id "c_half" counter)
            e-half {:op :stablehlo/constant :value 0.5 :outvars [c-half]}
            c-sqrt (gen-id "c_sqrt" counter)
            e-sqrt {:op :stablehlo/constant :value 0.7978845608 :outvars [c-sqrt]}
            c-poly (gen-id "c_poly" counter)
            e-poly {:op :stablehlo/constant :value 0.044715 :outvars [c-poly]}
            c-one (gen-id "c_one" counter)
            e-one {:op :stablehlo/constant :value 1.0 :outvars [c-one]}

            t-x2 (gen-id "t_x2" counter)
            e-x2 {:op :stablehlo/multiply :invars [in-var in-var] :outvars [t-x2]}
            t-x3 (gen-id "t_x3" counter)
            e-x3 {:op :stablehlo/multiply :invars [t-x2 in-var] :outvars [t-x3]}
            t-poly (gen-id "t_poly" counter)
            e-poly-mul {:op :stablehlo/multiply :invars [t-x3 c-poly] :outvars [t-poly]}
            t-inner (gen-id "t_inner" counter)
            e-inner {:op :stablehlo/add :invars [in-var t-poly] :outvars [t-inner]}
            t-scaled-inner (gen-id "t_scaled_inner" counter)
            e-sinner {:op :stablehlo/multiply :invars [t-inner c-sqrt] :outvars [t-scaled-inner]}
            t-tanh (gen-id "t_tanh" counter)
            e-tanh {:op :stablehlo/tanh :invars [t-scaled-inner] :outvars [t-tanh]}
            t-plus-one (gen-id "t_plus_one" counter)
            e-plus-one {:op :stablehlo/add :invars [t-tanh c-one] :outvars [t-plus-one]}
            t-half-x (gen-id "t_half_x" counter)
            e-half-x {:op :stablehlo/multiply :invars [in-var c-half] :outvars [t-half-x]}
            e-final {:op :stablehlo/multiply :invars [t-half-x t-plus-one] :outvars [out-var]}]
        (swap! eqns-atom conj e-half e-sqrt e-poly e-one e-x2 e-x3 e-poly-mul e-inner e-sinner e-tanh e-plus-one e-half-x e-final)
        out-var)

      softcap
      (let [cap (double softcap)
            c-cap (gen-id "c_cap" counter)
            c-cap-eqn {:op :stablehlo/constant :value cap :outvars [c-cap]}
            c-inv (gen-id "c_inv_cap" counter)
            c-inv-eqn {:op :stablehlo/constant :value (/ 1.0 cap) :outvars [c-inv]}
            scaled-var (gen-id "t_scaled_cap" counter)
            scale-eqn {:op :stablehlo/multiply :invars [in-var c-inv] :outvars [scaled-var]}
            tanh-var (gen-id "t_tanh_cap" counter)
            tanh-eqn {:op :stablehlo/tanh :invars [scaled-var] :outvars [tanh-var]}
            final-eqn {:op :stablehlo/multiply :invars [tanh-var c-cap] :outvars [out-var]}]
        (swap! eqns-atom conj c-cap-eqn c-inv-eqn scale-eqn tanh-eqn final-eqn)
        out-var)

      :else
      in-var)))

(defn- lower-binary-contraction!
  [eqns-atom counter head attrs lhs rhs _known-shapes var-dtypes final-out-var]
  (let [head-idxs (vec (rest head))
        lhs-name (first lhs)
        lhs-idxs (vec (rest lhs))
        rhs-name (first rhs)
        rhs-idxs (vec (rest rhs))
        {:keys [contracting batch lhs-free rhs-free]}
        (idx/partition-indices lhs-idxs rhs-idxs head-idxs)]
    (if (and (empty? contracting)
             (or (= (set lhs-idxs) (set head-idxs))
                 (= (set rhs-idxs) (set head-idxs))))
      ;; Elementwise or Broadcast product
      (let [primary-is-lhs? (= (set lhs-idxs) (set head-idxs))
            primary-name (if primary-is-lhs? lhs-name rhs-name)
            primary-idxs (if primary-is-lhs? lhs-idxs rhs-idxs)
            secondary-name (if primary-is-lhs? rhs-name lhs-name)
            secondary-idxs (if primary-is-lhs? rhs-idxs lhs-idxs)
            secondary-v (if (= (set primary-idxs) (set secondary-idxs))
                          (if (= primary-idxs secondary-idxs)
                            secondary-name
                            (let [rhs-perm (indices->dim-numbers secondary-idxs primary-idxs)
                                  out-r (gen-id "t_trans_r" counter)
                                  trans-r {:op :stablehlo/transpose :invars [secondary-name] :outvars [out-r] :attrs {:permutation rhs-perm}}]
                              (swap! eqns-atom conj trans-r)
                              out-r))
                          secondary-name)
            needs-perm? (not= primary-idxs head-idxs)
            has-post-act? (or (:scale attrs) (:act attrs) (:softcap attrs))
            mul-out-var (if (or needs-perm? has-post-act?)
                          (gen-id "t_mul" counter)
                          final-out-var)
            mul-eqn {:op :stablehlo/multiply
                     :invars [primary-name secondary-v]
                     :outvars [mul-out-var]}]
        (swap! eqns-atom conj mul-eqn)
        (let [perm-out-var
              (if needs-perm?
                (let [perm (indices->dim-numbers primary-idxs head-idxs)
                      out-v (if has-post-act? (gen-id "t_trans" counter) final-out-var)
                      trans-eqn {:op :stablehlo/transpose
                                 :invars [mul-out-var]
                                 :outvars [out-v]
                                 :attrs {:permutation perm}}]
                  (swap! eqns-atom conj trans-eqn)
                  out-v)
                mul-out-var)]
          (when has-post-act?
            (emit-post-activation! eqns-atom counter perm-out-var final-out-var attrs (get var-dtypes primary-name :f32)))))
      ;; Standard dot_general contraction
      (let [lhs-c-dims (indices->dim-numbers lhs-idxs contracting)
            rhs-c-dims (indices->dim-numbers rhs-idxs contracting)
            lhs-b-dims (indices->dim-numbers lhs-idxs batch)
            rhs-b-dims (indices->dim-numbers rhs-idxs batch)

            dot-attrs {:contracting_dims {:lhs lhs-c-dims :rhs rhs-c-dims}
                       :batch_dims {:lhs lhs-b-dims :rhs rhs-b-dims}}

            raw-dot-idxs (vec (concat batch lhs-free rhs-free))
            needs-perm? (not= raw-dot-idxs head-idxs)
            has-post-act? (or (:scale attrs) (:act attrs) (:softcap attrs))

            dot-out-var (if (or needs-perm? has-post-act?)
                          (gen-id "t_dot" counter)
                          final-out-var)

            dot-eqn {:op :stablehlo/dot_general
                     :invars [lhs-name rhs-name]
                     :outvars [dot-out-var]
                     :attrs dot-attrs}]
        (swap! eqns-atom conj dot-eqn)
        (let [perm-out-var
              (if needs-perm?
                (let [perm (indices->dim-numbers raw-dot-idxs head-idxs)
                      out-v (if has-post-act? (gen-id "t_trans" counter) final-out-var)
                      trans-eqn {:op :stablehlo/transpose
                                 :invars [dot-out-var]
                                 :outvars [out-v]
                                 :attrs {:permutation perm}}]
                  (swap! eqns-atom conj trans-eqn)
                  out-v)
                dot-out-var)
              _post-out-var
              (if has-post-act?
                (emit-post-activation! eqns-atom counter perm-out-var final-out-var attrs (get var-dtypes lhs-name :f32))
                perm-out-var)]
          nil)))))

(defn- lower-unary-equation!
  [eqns-atom counter head attrs term known-shapes var-dtypes final-out-var]
  (let [head-idxs (vec (rest head))
        term-name (first term)
        term-idxs (vec (rest term))
        has-post-act? (or (:scale attrs) (:act attrs) (:softcap attrs))]
    (cond
      ;; Direct copy or post-activation
      (= head-idxs term-idxs)
      (if has-post-act?
        (emit-post-activation! eqns-atom counter term-name final-out-var attrs (get var-dtypes term-name :f32))
        (let [target-shape (or (get known-shapes (first head))
                               (get known-shapes term-name))
              reshape-eqn {:op :stablehlo/reshape
                           :invars [term-name]
                           :outvars [final-out-var]
                           :attrs {:shape target-shape}}]
          (swap! eqns-atom conj reshape-eqn)))

      ;; Transpose / permutation
      (= (set head-idxs) (set term-idxs))
      (let [perm (indices->dim-numbers term-idxs head-idxs)
            trans-out (if has-post-act? (gen-id "t_trans" counter) final-out-var)
            trans-eqn {:op :stablehlo/transpose :invars [term-name] :outvars [trans-out] :attrs {:permutation perm}}]
        (swap! eqns-atom conj trans-eqn)
        (when has-post-act?
          (emit-post-activation! eqns-atom counter trans-out final-out-var attrs (get var-dtypes term-name :f32))))

      :else
      ;; Broadcast
      (let [target-shape (get known-shapes (first head))
            bcast-dims (indices->dim-numbers head-idxs term-idxs)
            bcast-out (if has-post-act? (gen-id "t_bcast" counter) final-out-var)
            bcast-eqn {:op :stablehlo/broadcast_in_dim
                       :invars [term-name]
                       :outvars [bcast-out]
                       :attrs {:broadcast_dimensions bcast-dims
                               :target_shape target-shape}}]
        (swap! eqns-atom conj bcast-eqn)
        (when has-post-act?
          (emit-post-activation! eqns-atom counter bcast-out final-out-var attrs (get var-dtypes term-name :f32)))))))

(defn- lower-gather! [eqns-atom counter _head table-term idx-term final-out-var known-shapes]
  (let [table-name (first table-term)
        idx-name (first idx-term)
        table-shape (get known-shapes table-name)
        hidden-dim (last table-shape)
        idx-shape (get known-shapes idx-name)
        expanded-idx-shape (conj (vec idx-shape) 1)
        reshaped-idx-var (gen-id "t_idx_reshape" counter)
        reshape-eqn {:op :stablehlo/reshape
                     :invars [idx-name]
                     :outvars [reshaped-idx-var]
                     :attrs {:shape expanded-idx-shape}}
        final-rank (count expanded-idx-shape)
        gather-eqn {:op :stablehlo/gather
                    :invars [table-name reshaped-idx-var]
                    :outvars [final-out-var]
                    :attrs {:offset_dims [(dec final-rank)]
                            :collapsed_slice_dims [0]
                            :start_index_map [0]
                            :index_vector_dim (dec final-rank)
                            :slice_sizes [1 hidden-dim]}}]
    (swap! eqns-atom conj reshape-eqn gather-eqn)))

(defn- lower-slice! [eqns-atom _head in-term attrs final-out-var]
  (let [in-name (first in-term)
        starts (or (:start_indices attrs) (:start attrs) [0 0 0])
        limits (or (:limit_indices attrs) (:limit attrs) [1 128 768])
        strides (or (:strides attrs) (vec (repeat (count starts) 1)))
        slice-eqn {:op :stablehlo/slice
                   :invars [in-name]
                   :outvars [final-out-var]
                   :attrs {:start_indices starts
                           :limit_indices limits
                           :strides strides}}]
    (swap! eqns-atom conj slice-eqn)))

(defn- lower-dynamic-slice! [eqns-atom _head in-term opt-starts attrs final-out-var]
  (let [in-name (first in-term)
        starts (or (when (vector? opt-starts) opt-starts)
                   (:start_indices attrs)
                   (:start-indices attrs)
                   [0 0 0])
        slice-sizes (or (:slice_sizes attrs) (:slice-sizes attrs) [1 1 1])
        slice-eqn {:op :stablehlo/dynamic_slice
                   :invars [in-name]
                   :outvars [final-out-var]
                   :attrs {:start_indices starts
                           :slice_sizes slice-sizes}}]
    (swap! eqns-atom conj slice-eqn)))

(defn- lower-dynamic-update-slice! [eqns-atom _head op-term up-term attrs final-out-var]
  (let [op-name (first op-term)
        up-name (first up-term)
        starts (or (:start_indices attrs) (:start-indices attrs) [0 0 0 0])
        update-eqn {:op :stablehlo/dynamic_update_slice
                    :invars [op-name up-name]
                    :outvars [final-out-var]
                    :attrs {:start_indices starts}}]
    (swap! eqns-atom conj update-eqn)))

(defn- lower-reshape! [eqns-atom head in-term attrs final-out-var]
  (let [in-name (first in-term)
        shape (or (:shape attrs) (vec (rest head)))
        reshape-eqn {:op :stablehlo/reshape
                     :invars [in-name]
                     :outvars [final-out-var]
                     :attrs {:shape shape}}]
    (swap! eqns-atom conj reshape-eqn)))

(defn- lower-layer-norm! [eqns-atom counter _head in-term gamma-term beta-term attrs final-out-var]
  (let [in-name (first in-term)
        gamma-name (first gamma-term)
        beta-name (first beta-term)
        eps (or (:eps attrs) 1e-5)
        mean-var (gen-id "ln_mean" counter)
        mean-eqn {:op :stablehlo/reduce_mean :invars [in-name] :outvars [mean-var] :attrs {:axes [-1] :keep_dims true}}
        diff-var (gen-id "ln_diff" counter)
        diff-eqn {:op :stablehlo/subtract :invars [in-name mean-var] :outvars [diff-var]}
        diff-sq-var (gen-id "ln_diff_sq" counter)
        diff-sq-eqn {:op :stablehlo/multiply :invars [diff-var diff-var] :outvars [diff-sq-var]}
        var-var (gen-id "ln_var" counter)
        var-eqn {:op :stablehlo/reduce_mean :invars [diff-sq-var] :outvars [var-var] :attrs {:axes [-1] :keep_dims true}}
        c-eps (gen-id "ln_eps" counter)
        c-eps-eqn {:op :stablehlo/constant :value (double eps) :outvars [c-eps]}
        var-eps-var (gen-id "ln_var_eps" counter)
        var-eps-eqn {:op :stablehlo/add :invars [var-var c-eps] :outvars [var-eps-var]}
        std-var (gen-id "ln_std" counter)
        std-eqn {:op :stablehlo/sqrt :invars [var-eps-var] :outvars [std-var]}
        xhat-var (gen-id "ln_xhat" counter)
        xhat-eqn {:op :stablehlo/divide :invars [diff-var std-var] :outvars [xhat-var]}
        scaled-var (gen-id "ln_scaled" counter)
        scaled-eqn {:op :stablehlo/multiply :invars [xhat-var gamma-name] :outvars [scaled-var]}
        final-eqn {:op :stablehlo/add :invars [scaled-var beta-name] :outvars [final-out-var]}]
    (swap! eqns-atom conj mean-eqn diff-eqn diff-sq-eqn var-eqn c-eps-eqn var-eps-eqn std-eqn xhat-eqn scaled-eqn final-eqn)))

(defn- lower-causal-softmax!
  ([eqns-atom counter in-term attrs final-out-var known-shapes]
   (lower-causal-softmax! eqns-atom counter in-term attrs final-out-var known-shapes :f32))
  ([eqns-atom counter in-term attrs final-out-var known-shapes dtype]
   (let [in-name (first in-term)
         orig-dtype (or dtype :f32)
         scores-shape (get known-shapes in-name [1 12 128 128])
         batch (nth scores-shape 0 1)
         num-heads (nth scores-shape 1 12)
         q-len (nth scores-shape 2 128)
         kv-len (nth scores-shape 3 128)
         window (or (:sliding-window attrs) (:window-size attrs))
         pos-var (:pos attrs)
         needs-f32? (not= orig-dtype :f32)
         f32-in-var (if needs-f32? (gen-id "s_f32" counter) in-name)
         conv-in-eqn (when needs-f32?
                       {:op :stablehlo/convert :invars [in-name] :outvars [f32-in-var] :attrs {:target_dtype :f32}})
         masked-var (gen-id "t_masked" counter)]
     (when needs-f32? (swap! eqns-atom conj conv-in-eqn))
     (if pos-var
       (let [pos-scalar (gen-id "pos_s" counter)
             pos-s-eqn {:op :stablehlo/reshape :invars [pos-var] :outvars [pos-scalar] :attrs {:shape []}}
             pos-4d (gen-id "pos_4d" counter)
             pos-4d-eqn {:op :stablehlo/broadcast_in_dim :invars [pos-scalar] :outvars [pos-4d] :attrs {:broadcast_dimensions [] :target_shape [1 1 1 kv-len]}}
             iota-1d (gen-id "iota_1d" counter)
             iota-1d-eqn {:op :stablehlo/iota :outvars [iota-1d] :attrs {:len kv-len :dtype :i32 :iota_dimension 0}}
             iota-4d (gen-id "iota_4d" counter)
             iota-4d-eqn {:op :stablehlo/broadcast_in_dim :invars [iota-1d] :outvars [iota-4d] :attrs {:broadcast_dimensions [3] :target_shape [1 1 1 kv-len]}}
             cmp-fut (gen-id "cmp_fut" counter)
             cmp-fut-eqn {:op :stablehlo/compare :invars [iota-4d pos-4d] :outvars [cmp-fut] :attrs {:comparison_direction "GT"}}
             _ (swap! eqns-atom conj pos-s-eqn pos-4d-eqn iota-1d-eqn iota-4d-eqn cmp-fut-eqn)
             cmp-final (if window
                         (let [c-win (gen-id "c_win" counter)
                               c-win-eqn {:op :stablehlo/constant :value (int window) :type [:tensor [] :i32] :outvars [c-win]}
                               p-sub-win (gen-id "p_sub_w" counter)
                               p-sub-eqn {:op :stablehlo/subtract :invars [pos-scalar c-win] :outvars [p-sub-win]}
                               c-one (gen-id "c_one" counter)
                               c-one-eqn {:op :stablehlo/constant :value 1 :type [:tensor [] :i32] :outvars [c-one]}
                               min-p (gen-id "min_p" counter)
                               min-p-eqn {:op :stablehlo/add :invars [p-sub-win c-one] :outvars [min-p]}
                               min-p-4d (gen-id "min_p_4d" counter)
                               min-p-4d-eqn {:op :stablehlo/broadcast_in_dim :invars [min-p] :outvars [min-p-4d] :attrs {:broadcast_dimensions [] :target_shape [1 1 1 kv-len]}}
                               cmp-old (gen-id "cmp_old" counter)
                               cmp-old-eqn {:op :stablehlo/compare :invars [iota-4d min-p-4d] :outvars [cmp-old] :attrs {:comparison_direction "LT"}}
                               cmp-comb (gen-id "cmp_comb" counter)
                               cmp-comb-eqn {:op :stablehlo/or :invars [cmp-fut cmp-old] :outvars [cmp-comb]}]
                           (swap! eqns-atom conj c-win-eqn p-sub-eqn c-one-eqn min-p-eqn min-p-4d-eqn cmp-old-eqn cmp-comb-eqn)
                           cmp-comb)
                         cmp-fut)
             c-neg (gen-id "c_neg" counter)
             c-neg-eqn {:op :stablehlo/constant :value -10000.0 :type [:tensor [] :f32] :outvars [c-neg]}
             c-neg-4d (gen-id "c_neg_4d" counter)
             c-neg-4d-eqn {:op :stablehlo/broadcast_in_dim :invars [c-neg] :outvars [c-neg-4d] :attrs {:broadcast_dimensions [] :target_shape [1 1 1 kv-len]}}
             c-zero (gen-id "c_zero" counter)
             c-zero-eqn {:op :stablehlo/constant :value 0.0 :type [:tensor [] :f32] :outvars [c-zero]}
             c-zero-4d (gen-id "c_zero_4d" counter)
             c-zero-4d-eqn {:op :stablehlo/broadcast_in_dim :invars [c-zero] :outvars [c-zero-4d] :attrs {:broadcast_dimensions [] :target_shape [1 1 1 kv-len]}}
             dyn-mask (gen-id "c_dyn_mask" counter)
             dyn-mask-eqn {:op :stablehlo/select :invars [cmp-final c-neg-4d c-zero-4d] :outvars [dyn-mask]}
             mask-broad (gen-id "c_mask_bcast" counter)
             mask-broad-eqn {:op :stablehlo/broadcast_in_dim :invars [dyn-mask] :outvars [mask-broad] :attrs {:broadcast_dimensions [0 1 2 3] :target_shape [batch num-heads q-len kv-len]}}
             masked-eqn {:op :stablehlo/add :invars [f32-in-var mask-broad] :outvars [masked-var]}]
         (swap! eqns-atom conj c-neg-eqn c-neg-4d-eqn c-zero-eqn c-zero-4d-eqn dyn-mask-eqn mask-broad-eqn masked-eqn))
       (let [mask-rows (vec (for [i (range q-len)]
                              (vec (for [j (range kv-len)]
                                     (if (or (> j i) (and window (>= (- i j) (long window))))
                                       -10000.0
                                       0.0)))))
             c-mask (gen-id "c_mask" counter)
             c-mask-eqn {:op :stablehlo/constant
                         :value [[mask-rows]]
                         :type [:tensor [1 1 q-len kv-len] :f32]
                         :outvars [c-mask]}
             masked-eqn {:op :stablehlo/add :invars [f32-in-var c-mask] :outvars [masked-var]}]
         (swap! eqns-atom conj c-mask-eqn masked-eqn)))
     (let [max-var (gen-id "t_smax" counter)
           max-eqn {:op :stablehlo/reduce_max :invars [masked-var] :outvars [max-var] :attrs {:axes [-1] :keep_dims true}}
           diff-var (gen-id "t_sdiff" counter)
           diff-eqn {:op :stablehlo/subtract :invars [masked-var max-var] :outvars [diff-var]}
           exp-var (gen-id "t_sexp" counter)
           exp-eqn {:op :stablehlo/exp :invars [diff-var] :outvars [exp-var]}
           sum-var (gen-id "t_ssum" counter)
           sum-eqn {:op :stablehlo/reduce_sum :invars [exp-var] :outvars [sum-var] :attrs {:axes [-1] :keep_dims true}}
           f32-div-var (if needs-f32? (gen-id "t_sdiv" counter) final-out-var)
           div-eqn {:op :stablehlo/divide :invars [exp-var sum-var] :outvars [f32-div-var]}
           conv-out-eqn (when needs-f32?
                          {:op :stablehlo/convert :invars [f32-div-var] :outvars [final-out-var] :attrs {:target_dtype orig-dtype}})]
       (swap! eqns-atom conj max-eqn diff-eqn exp-eqn sum-eqn div-eqn)
       (when needs-f32? (swap! eqns-atom conj conv-out-eqn))))))

(defn- lower-rms-norm! [eqns-atom counter _head in-term weight-term attrs final-out-var]
  (let [in-name (first in-term)
        weight-name (when weight-term (first weight-term))
        gemma? (:gemma? attrs)
        eps (or (:eps attrs) 1e-5)
        sq-var (gen-id "rms_sq" counter)
        sq-eqn {:op :stablehlo/multiply :invars [in-name in-name] :outvars [sq-var]}
        mean-var (gen-id "rms_mean" counter)
        mean-eqn {:op :stablehlo/reduce_mean :invars [sq-var] :outvars [mean-var] :attrs {:axes [-1] :keep_dims true}}
        c-eps (gen-id "rms_eps" counter)
        c-eps-eqn {:op :stablehlo/constant :value (double eps) :outvars [c-eps]}
        mean-eps-var (gen-id "rms_mean_eps" counter)
        mean-eps-eqn {:op :stablehlo/add :invars [mean-var c-eps] :outvars [mean-eps-var]}
        rsqrt-var (gen-id "rms_rsqrt" counter)
        rsqrt-eqn {:op :stablehlo/rsqrt :invars [mean-eps-var] :outvars [rsqrt-var]}
        xhat-var (if weight-name (gen-id "rms_xhat" counter) final-out-var)
        xhat-eqn {:op :stablehlo/multiply :invars [in-name rsqrt-var] :outvars [xhat-var]}]
    (swap! eqns-atom conj sq-eqn mean-eqn c-eps-eqn mean-eps-eqn rsqrt-eqn xhat-eqn)
    (when weight-name
      (let [w-var (if gemma?
                    (let [c-one (gen-id "rms_one" counter)
                          c-one-eqn {:op :stablehlo/constant :value 1.0 :outvars [c-one]}
                          w-plus-one (gen-id "rms_w1" counter)
                          add-one-eqn {:op :stablehlo/add :invars [weight-name c-one] :outvars [w-plus-one]}]
                      (swap! eqns-atom conj c-one-eqn add-one-eqn)
                      w-plus-one)
                    weight-name)
            scaled-eqn {:op :stablehlo/multiply :invars [xhat-var w-var] :outvars [final-out-var]}]
        (swap! eqns-atom conj scaled-eqn)))))

(defn- lower-rope!
  ([eqns-atom counter head in-term attrs final-out-var known-shapes]
   (lower-rope! eqns-atom counter head in-term attrs final-out-var known-shapes :f32))
  ([eqns-atom counter _head in-term attrs final-out-var known-shapes dtype]
   (let [in-name (first in-term)
         dtype (or dtype :f32)
         shape (get known-shapes in-name [1 128 576])
         batch (nth shape 0 1)
         seq-len (nth shape 1 128)
         total-dim (nth shape 2 576)
         head-dim (long (or (:head-dim attrs) 64))
         n-heads (quot total-dim head-dim)
         rope-prop (double (or (:rope-proportion attrs) (:partial-rotary-factor attrs) 1.0))
         rot-dim (long (* head-dim rope-prop))
         partial? (< rot-dim head-dim)
         effective-rot-dim (if partial? rot-dim head-dim)
         half-dim (quot effective-rot-dim 2)
         theta (double (or (:theta attrs) (:theta-base attrs) 10000.0))

         dynamic? (and (= seq-len 1) (:pos attrs))
         pos-var (when dynamic? (:pos attrs))
         table-len (if dynamic? (long (or (:max-seq-len attrs) 2048)) seq-len)

         freqs (vec (for [i (range half-dim)]
                      (Math/pow theta (/ (* -2.0 i) (double effective-rot-dim)))))
         cos-rows (vec (for [idx (range table-len)]
                         (let [pos idx
                               half (vec (for [i (range half-dim)]
                                           (Math/cos (* (double pos) (nth freqs i)))))]
                           (vec (concat half half)))))
         sin-rows (vec (for [idx (range table-len)]
                         (let [pos idx
                               half (vec (for [i (range half-dim)]
                                           (Math/sin (* (double pos) (nth freqs i)))))]
                           (vec (concat half half)))))

         r4d-var (gen-id "rope_r4d" counter)
         r4d-eqn {:op :stablehlo/reshape :invars [in-name] :outvars [r4d-var] :attrs {:shape [batch seq-len n-heads head-dim]}}

         trans-var (gen-id "rope_trans" counter)
         trans-eqn {:op :stablehlo/transpose :invars [r4d-var] :outvars [trans-var] :attrs {:permutation [0 2 1 3]}}

         x-rope-var (if partial? (gen-id "rope_x_rot" counter) trans-var)
         x-pass-var (when partial? (gen-id "rope_x_pass" counter))
         slice-rot-eqn (when partial?
                         {:op :stablehlo/slice
                          :invars [trans-var]
                          :outvars [x-rope-var]
                          :attrs {:start_indices [0 0 0 0]
                                  :limit_indices [batch n-heads seq-len effective-rot-dim]
                                  :strides [1 1 1 1]}})
         slice-pass-eqn (when partial?
                          {:op :stablehlo/slice
                           :invars [trans-var]
                           :outvars [x-pass-var]
                           :attrs {:start_indices [0 0 0 effective-rot-dim]
                                   :limit_indices [batch n-heads seq-len head-dim]
                                   :strides [1 1 1 1]}})

         x1-var (gen-id "rope_x1" counter)
         x1-eqn {:op :stablehlo/slice
                 :invars [x-rope-var]
                 :outvars [x1-var]
                 :attrs {:start_indices [0 0 0 0]
                         :limit_indices [batch n-heads seq-len half-dim]
                         :strides [1 1 1 1]}}

         x2-var (gen-id "rope_x2" counter)
         x2-eqn {:op :stablehlo/slice
                 :invars [x-rope-var]
                 :outvars [x2-var]
                 :attrs {:start_indices [0 0 0 half-dim]
                         :limit_indices [batch n-heads seq-len effective-rot-dim]
                         :strides [1 1 1 1]}}

         neg-x2-var (gen-id "rope_neg_x2" counter)
         neg-x2-eqn {:op :stablehlo/negate :invars [x2-var] :outvars [neg-x2-var]}

         rot-var (gen-id "rope_rot" counter)
         rot-eqn {:op :stablehlo/concatenate :invars [neg-x2-var x1-var] :outvars [rot-var] :attrs {:dimension 3}}

         c-cos-table (gen-id "rope_cos" counter)
         c-cos-eqn {:op :stablehlo/constant
                    :value [[cos-rows]]
                    :type [:tensor [1 1 table-len effective-rot-dim] dtype]
                    :outvars [c-cos-table]}

         c-sin-table (gen-id "rope_sin" counter)
         c-sin-eqn {:op :stablehlo/constant
                    :value [[sin-rows]]
                    :type [:tensor [1 1 table-len effective-rot-dim] dtype]
                    :outvars [c-sin-table]}

         [c-cos-var c-sin-var]
         (if dynamic?
           (let [cos-sl (gen-id "rope_cos_sl" counter)
                 cos-sl-eqn {:op :stablehlo/dynamic_slice
                             :invars [c-cos-table]
                             :outvars [cos-sl]
                             :attrs {:slice_sizes [1 1 1 effective-rot-dim]
                                     :start_indices [0 0 pos-var 0]}}
                 sin-sl (gen-id "rope_sin_sl" counter)
                 sin-sl-eqn {:op :stablehlo/dynamic_slice
                             :invars [c-sin-table]
                             :outvars [sin-sl]
                             :attrs {:slice_sizes [1 1 1 effective-rot-dim]
                                     :start_indices [0 0 pos-var 0]}}
                 cos-bc (gen-id "rope_cos_bc" counter)
                 cos-bc-eqn {:op :stablehlo/broadcast_in_dim
                             :invars [cos-sl]
                             :outvars [cos-bc]
                             :attrs {:broadcast_dimensions [0 1 2 3]
                                     :target_shape [batch n-heads 1 effective-rot-dim]}}
                 sin-bc (gen-id "rope_sin_bc" counter)
                 sin-bc-eqn {:op :stablehlo/broadcast_in_dim
                             :invars [sin-sl]
                             :outvars [sin-bc]
                             :attrs {:broadcast_dimensions [0 1 2 3]
                                     :target_shape [batch n-heads 1 effective-rot-dim]}}]
             (swap! eqns-atom conj c-cos-eqn c-sin-eqn cos-sl-eqn sin-sl-eqn cos-bc-eqn sin-bc-eqn)
             [cos-bc sin-bc])
           (do
             (swap! eqns-atom conj c-cos-eqn c-sin-eqn)
             [c-cos-table c-sin-table]))

         x-cos-var (gen-id "rope_x_cos" counter)
         x-cos-eqn {:op :stablehlo/multiply :invars [x-rope-var c-cos-var] :outvars [x-cos-var]}

         x-sin-var (gen-id "rope_x_sin" counter)
         x-sin-eqn {:op :stablehlo/multiply :invars [rot-var c-sin-var] :outvars [x-sin-var]}

         add-rot-var (if partial? (gen-id "rope_res_rot" counter) (gen-id "rope_res" counter))
         add-rot-eqn {:op :stablehlo/add :invars [x-cos-var x-sin-var] :outvars [add-rot-var]}

         add-var (if partial? (gen-id "rope_res" counter) add-rot-var)
         concat-pass-eqn (when partial?
                           {:op :stablehlo/concatenate :invars [add-rot-var x-pass-var] :outvars [add-var] :attrs {:dimension 3}})

         back-trans-var (gen-id "rope_back_trans" counter)
         back-trans-eqn {:op :stablehlo/transpose :invars [add-var] :outvars [back-trans-var] :attrs {:permutation [0 2 1 3]}}

         final-eqn {:op :stablehlo/reshape :invars [back-trans-var] :outvars [final-out-var] :attrs {:shape [batch seq-len total-dim]}}]
     (swap! eqns-atom conj r4d-eqn trans-eqn)
     (when partial?
       (swap! eqns-atom conj slice-rot-eqn slice-pass-eqn))
     (swap! eqns-atom conj x1-eqn x2-eqn neg-x2-eqn rot-eqn x-cos-eqn x-sin-eqn add-rot-eqn)
     (when partial?
       (swap! eqns-atom conj concat-pass-eqn))
     (swap! eqns-atom conj back-trans-eqn final-eqn))))

(defn ast->graph
  "Compiles a Tensor Logic Hiccup AST into a validated EDN SSA graph for OpenXLA compilation.
   Pipeline: expand -> prune (DCE) -> unify shapes -> lower to dot_general & StableHLO ops."
  [graph-name invars ast target-heads]
  (let [targets (if (set? target-heads) target-heads (set target-heads))
        ;; 1. Expand containers & decompose multi-term contractions
        expanded (expand/expand-ast {} ast)
        ;; 2. Backward-chaining DCE
        pruned (dce/prune-ast expanded targets)
        ;; 3. Shape unification
        in-shapes (into {} (map (fn [[v [_ shape _]]] [v (vec shape)]) invars))
        in-dtypes (into {} (map (fn [[v [_ _ dt]]] [v dt]) invars))
        default-dtype (or (get in-dtypes :embed_tokens) (get in-dtypes :final_norm_w) :f32)
        const-dtypes (into {} (keep (fn [eqn]
                                      (when (= (first eqn) :constant)
                                        (let [h (second eqn)
                                              h-name (if (vector? h) (first h) h)
                                              attrs (ast/attrs eqn)
                                              val (get attrs :value false)
                                              t (or (:type attrs) (if (boolean? val) [:tensor [] :i1] [:tensor [] :f32]))]
                                          [h-name (last t)])))
                                    pruned))
        known-dtypes (merge in-dtypes const-dtypes)
        known-shapes (shape/unify-shapes in-shapes pruned)

        eqns-atom (atom [])
        counter (atom 0)

        ;; 4. Track occurrences of heads for inline implicit accumulation
        head-total-counts (frequencies (map ast/head pruned))
        accum-state (atom {})]

    ;; Lower equations in topological order
    (doseq [eqn pruned]
      (let [op (first eqn)
            head (ast/head eqn)
            h-name (first head)
            attrs (ast/attrs eqn)
            body (ast/body-terms eqn)
            total (get head-total-counts head 1)
            seen (get-in @accum-state [head :seen] 0)
            idx (inc seen)
            is-accum? (> total 1)
            term-var (if is-accum?
                       (gen-id (str (name h-name) "_term" idx) counter)
                       h-name)
            final-var term-var]
        ;; Lower the underlying equation to write into final-var
        (cond
          (ast/eqn? eqn)
          (cond
            (= (count body) 2)
            (lower-binary-contraction! eqns-atom counter head attrs (first body) (second body)
                                       known-shapes in-dtypes final-var)

            (= (count body) 1)
            (lower-unary-equation! eqns-atom counter head attrs (first body)
                                   known-shapes in-dtypes final-var)

            :else
            (throw (ex-info "Unsupported contraction body arity after expansion" {:equation eqn})))

          (= op :gather)
          (lower-gather! eqns-atom counter head (first body) (second body) final-var known-shapes)

          (= op :slice)
          (lower-slice! eqns-atom head (first body) attrs final-var)

          (= op :dynamic-slice)
          (lower-dynamic-slice! eqns-atom head (first body) (second body) attrs final-var)

          (= op :dynamic-update-slice)
          (lower-dynamic-update-slice! eqns-atom head (first body) (second body) attrs final-var)

          (= op :reshape)
          (lower-reshape! eqns-atom head (first body) attrs final-var)

          (= op :layer-norm)
          (lower-layer-norm! eqns-atom counter head (first body) (second body) (nth body 2) attrs final-var)

          (= op :causal-softmax)
          (lower-causal-softmax! eqns-atom counter (first body) attrs final-var known-shapes default-dtype)

          (or (= op :rms-norm) (= op :gemma-rms-norm))
          (lower-rms-norm! eqns-atom counter head (first body) (second body)
                           (if (= op :gemma-rms-norm) (assoc attrs :gemma? true) attrs)
                           final-var)

          (= op :rope)
          (lower-rope! eqns-atom counter head (first body) attrs final-var known-shapes default-dtype)

          (= op :while)
          (let [out-spec (second eqn)
                in-spec (nth eqn 2)
                out-vars (mapv #(if (vector? %) (first %) %) (if (vector? out-spec) out-spec [out-spec]))
                in-vars (mapv #(if (vector? %) (first %) %) (if (vector? in-spec) in-spec [in-spec]))
                cond-node (or (get attrs :cond)
                              (get attrs :cond-node)
                              (get attrs :cond-ast)
                              (first (filter #(and (vector? %) (= (first %) :cond)) (drop 3 eqn))))
                cond-graph (when cond-node
                             (let [cond-head (second cond-node)
                                   cond-out (if (vector? cond-head) (first cond-head) cond-head)
                                   cond-children (filter vector?
                                                         (if (map? (nth cond-node 2 nil))
                                                           (drop 3 cond-node)
                                                           (drop 2 cond-node)))
                                   cond-arg-names (or (:cond-args attrs)
                                                      (:args (ast/attrs cond-node))
                                                      in-vars)
                                   cond-invars (mapv (fn [arg-name in-var]
                                                       (let [sh (or (get in-shapes in-var)
                                                                    (get known-shapes in-var)
                                                                    [])
                                                             dt (or (get known-dtypes in-var)
                                                                    (get in-dtypes in-var)
                                                                    (when (re-find #"(?:step|pos|count|max)" (name in-var)) :i32)
                                                                    (when (re-find #"(?:stopped|bool|flag|false|true)" (name in-var)) :i1)
                                                                    :i32)]
                                                         [arg-name [:tensor sh dt]]))
                                                     cond-arg-names
                                                     in-vars)
                                   cond-ast (vec (into [:block {}] cond-children))]
                               (ast->graph "cond" cond-invars cond-ast [cond-out])))
                body-node (or (get attrs :body)
                              (get attrs :body-node)
                              (get attrs :body-ast)
                              (first (filter #(and (vector? %) (= (first %) :body)) (drop 3 eqn))))
                body-graph (when body-node
                             (let [body-head (second body-node)
                                   body-outs (if (vector? body-head)
                                               (mapv #(if (vector? %) (first %) %) body-head)
                                               [body-head])
                                   body-children (filter vector?
                                                         (if (map? (nth body-node 2 nil))
                                                           (drop 3 body-node)
                                                           (drop 2 body-node)))
                                   body-arg-names (or (:body-args attrs)
                                                      (:args (ast/attrs body-node))
                                                      in-vars)
                                   carry-invars (mapv (fn [arg-name in-var]
                                                        (let [sh (or (get in-shapes in-var)
                                                                     (get known-shapes in-var)
                                                                     [])
                                                              dt (or (get known-dtypes in-var)
                                                                     (get in-dtypes in-var)
                                                                     (when (re-find #"(?:step|pos|count|max)" (name in-var)) :i32)
                                                                     (when (re-find #"(?:stopped|bool|flag|false|true)" (name in-var)) :i1)
                                                                     :i32)]
                                                          [arg-name [:tensor sh dt]]))
                                                      body-arg-names
                                                      in-vars)
                                   all-body-invars (into carry-invars invars)
                                   body-ast (vec (into [:block {}] body-children))
                                   bg (ast->graph "body" all-body-invars body-ast body-outs)]
                               (assoc bg :carry-invars carry-invars)))
                attrs (cond-> attrs
                        cond-graph (assoc :cond-graph cond-graph)
                        body-graph (assoc :body-graph body-graph))]
            (swap! eqns-atom conj {:op :stablehlo/while
                                   :invars in-vars
                                   :outvars out-vars
                                   :attrs attrs}))

          (or (= op :+) (= op :add))
          (let [h-name (if (vector? head) (first head) head)]
            (swap! eqns-atom conj {:op :stablehlo/add
                                   :invars [(first (first body)) (first (second body))]
                                   :outvars [h-name]}))

          (or (= op :-) (= op :subtract))
          (let [h-name (if (vector? head) (first head) head)]
            (swap! eqns-atom conj {:op :stablehlo/subtract
                                   :invars [(first (first body)) (first (second body))]
                                   :outvars [h-name]}))

          (or (= op :*) (= op :multiply))
          (let [h-name (if (vector? head) (first head) head)]
            (swap! eqns-atom conj {:op :stablehlo/multiply
                                   :invars [(first (first body)) (first (second body))]
                                   :outvars [h-name]}))

          (= op :convert)
          (let [h-name (if (vector? head) (first head) head)
                target-dtype (or (:target-dtype attrs) (:target_dtype attrs) :f32)]
            (swap! eqns-atom conj {:op :stablehlo/convert
                                   :invars [(first (first body))]
                                   :outvars [h-name]
                                   :attrs {:target_dtype target-dtype}}))

          (= op :argmax)
          (let [h-name (if (vector? head) (first head) head)
                axis (or (:axis attrs) (:dimension attrs) 1)]
            (swap! eqns-atom conj {:op :stablehlo/argmax
                                   :invars [(first (first body))]
                                   :outvars [h-name]
                                   :attrs {:axis axis}}))

          (= op :compare)
          (let [h-name (if (vector? head) (first head) head)
                dir (or (:direction attrs) (:comparison_direction attrs) "LT")
                dir-str (if (keyword? dir) (name dir) (str dir))]
            (swap! eqns-atom conj {:op :stablehlo/compare
                                   :invars [(first (first body)) (first (second body))]
                                   :outvars [h-name]
                                   :attrs {:comparison_direction dir-str}}))

          (= op :not)
          (let [h-name (if (vector? head) (first head) head)]
            (swap! eqns-atom conj {:op :stablehlo/not
                                   :invars [(first (first body))]
                                   :outvars [h-name]}))

          (or (= op :and) (= op :or))
          (let [h-name (if (vector? head) (first head) head)
                op-kw (if (= op :and) :stablehlo/and :stablehlo/or)]
            (swap! eqns-atom conj {:op op-kw
                                   :invars [(first (first body)) (first (second body))]
                                   :outvars [h-name]}))

          (= op :cond)
          (let [cond-children (filter vector?
                                      (if (map? (nth eqn 2 nil))
                                        (drop 3 eqn)
                                        (drop 2 eqn)))]
            (doseq [child cond-children]
              (let [c-op (first child)
                    c-head (second child)
                    c-body (ast/body-terms child)
                    c-attrs (ast/attrs child)]
                (case c-op
                  :compare
                  (let [h (if (vector? c-head) (first c-head) c-head)
                        dir (or (:direction c-attrs) (:comparison_direction c-attrs) "LT")
                        dir-str (if (keyword? dir) (name dir) (str dir))]
                    (swap! eqns-atom conj {:op :stablehlo/compare
                                           :invars [(first (first c-body)) (first (second c-body))]
                                           :outvars [h]
                                           :attrs {:comparison_direction dir-str}}))
                  :not
                  (let [h (if (vector? c-head) (first c-head) c-head)]
                    (swap! eqns-atom conj {:op :stablehlo/not
                                           :invars [(first (first c-body))]
                                           :outvars [h]}))
                  :and
                  (let [h (if (vector? c-head) (first c-head) c-head)]
                    (swap! eqns-atom conj {:op :stablehlo/and
                                           :invars [(first (first c-body)) (first (second c-body))]
                                           :outvars [h]}))
                  :or
                  (let [h (if (vector? c-head) (first c-head) c-head)]
                    (swap! eqns-atom conj {:op :stablehlo/or
                                           :invars [(first (first c-body)) (first (second c-body))]
                                           :outvars [h]}))
                  nil))))

          (= op :constant)
          (let [h (second eqn)
                h-name (if (vector? h) (first h) h)
                val (get attrs :value false)
                val-t (or (:type attrs) (if (boolean? val) [:tensor [] :i1] [:tensor [] :f32]))]
            (swap! eqns-atom conj {:op :stablehlo/constant
                                   :value val
                                   :type val-t
                                   :outvars [h-name]}))

          :else
          (throw (ex-info "Unknown AST equation or lowering hook" {:equation eqn :op op})))

        ;; If multi-term accumulation, emit inline addition immediately when idx > 1
        (when is-accum?
          (if (= idx 1)
            (swap! accum-state assoc head {:seen 1 :current-var term-var})
            (let [prev-var (get-in @accum-state [head :current-var])
                  is-last? (= idx total)
                  out-var (if is-last? h-name (gen-id (str (name h-name) "_acc" idx) counter))
                  add-eqn {:op :stablehlo/add :invars [prev-var term-var] :outvars [out-var]}]
              (swap! eqns-atom conj add-eqn)
              (swap! accum-state assoc head {:seen idx :current-var out-var}))))))

    (let [outvars (if (sequential? target-heads) (vec target-heads) (vec targets))
          graph {:name graph-name
                 :invars invars
                 :outvars outvars
                 :eqns @eqns-atom}]
      (shlo/validate-graph graph))))
