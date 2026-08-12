(ns clj-xla.nn.attention
  "Causal self-attention, GQA, and RoPE positioning mechanisms."
  (:refer-clojure :exclude [+ * - /])
  (:require [clj-xla.tensor :refer [+ * - / broadcast-in-dim concatenate convert cos sin dynamic-update-slice exp dot-general matmul maximum reduce-max reduce-sum reshape slice tanh transpose]]))

(defn linear
  "Linear projection layer: x @ w + b."
  [x w b]
  (let [h (matmul x w)]
    (if (some? b) (+ h b) h)))

(defn- extract-pos-int [pos]
  (cond
    (nil? pos) 0
    (number? pos) (long pos)
    (sequential? pos) (long (or (first pos) 0))
    :else 0))

(defn- generate-rope-freq-tensors
  ([seq-len head-dim theta-base] (generate-rope-freq-tensors seq-len head-dim theta-base 0 head-dim 1.0))
  ([seq-len head-dim theta-base pos-offset] (generate-rope-freq-tensors seq-len head-dim theta-base pos-offset head-dim 1.0))
  ([seq-len head-dim theta-base pos-offset full-dim] (generate-rope-freq-tensors seq-len head-dim theta-base pos-offset full-dim 1.0))
  ([seq-len head-dim theta-base pos-offset full-dim rope-proportion]
   (let [f-dim (or full-dim head-dim)
         t-base (or theta-base 10000.0)
         rope-prop (double (or rope-proportion 1.0))
         half-dim (quot head-dim 2)
         rope-angles (long (clojure.core/* rope-prop half-dim))]
     (if (clj-xla.tensor/tracer? pos-offset)
       (let [freqs-vec (vec (for [i (range half-dim)]
                              (if (clojure.core/< i rope-angles)
                                (Math/pow t-base (clojure.core// (clojure.core/* -2.0 i) (double f-dim)))
                                0.0)))
             freqs-const (clj-xla.tensor/emit-constant! [[[[freqs-vec]]]] [:tensor [1 1 1 half-dim] :f32])
             pos-f32 (convert pos-offset :f32)
             pos-shape (second (:type pos-f32))
             s-len (if (= (count pos-shape) 2) (second pos-shape) (first pos-shape))
             pos-4d (reshape pos-f32 [1 1 s-len 1])
             angles (* pos-4d freqs-const)
             cos-half (cos angles)
             sin-half (sin angles)
             cos-t (concatenate [cos-half cos-half] 3)
             sin-t (concatenate [sin-half sin-half] 3)]
         [cos-t sin-t])
       (let [p-off (extract-pos-int pos-offset)
             freqs (vec (for [i (range half-dim)]
                          (if (clojure.core/< i rope-angles)
                            (Math/pow t-base (clojure.core// (clojure.core/* -2.0 i) (double f-dim)))
                            0.0)))
             cos-rows (vec (for [idx (range seq-len)]
                             (let [pos (clojure.core/+ p-off idx)
                                   half (vec (for [i (range half-dim)]
                                               (Math/cos (clojure.core/* (double pos) (nth freqs i)))))]
                               (vec (clojure.core/concat half half)))))
             sin-rows (vec (for [idx (range seq-len)]
                             (let [pos (clojure.core/+ p-off idx)
                                   half (vec (for [i (range half-dim)]
                                               (Math/sin (clojure.core/* (double pos) (nth freqs i)))))]
                               (vec (clojure.core/concat half half)))))]
         [(clj-xla.tensor/emit-constant! [[cos-rows]] [:tensor [1 1 seq-len head-dim] :f32])
          (clj-xla.tensor/emit-constant! [[sin-rows]] [:tensor [1 1 seq-len head-dim] :f32])])))))

(defn- apply-rope-single
  ([x pos-ids head-dim theta-base]
   (apply-rope-single x pos-ids head-dim (or theta-base 10000.0) nil head-dim 1.0))
  ([x pos-ids head-dim theta-base rotary-dim]
   (apply-rope-single x pos-ids head-dim (or theta-base 10000.0) rotary-dim head-dim 1.0))
  ([x pos-ids head-dim theta-base rotary-dim full-dim]
   (apply-rope-single x pos-ids head-dim theta-base rotary-dim full-dim 1.0))
  ([x pos-ids head-dim theta-base rotary-dim full-dim rope-proportion]
   (let [x-type (:type x)
         [_ shape dtype] x-type
         is-3d (= (count shape) 3)
         t-base (or theta-base 10000.0)]
     (if is-3d
       (let [[batch seq-len q-dim] shape
             h-dim (or head-dim 64)
             n-heads (quot q-dim h-dim)
             x-4d (transpose (reshape x [batch seq-len n-heads h-dim]) [0 2 1 3])
             res-4d (apply-rope-single x-4d pos-ids h-dim t-base rotary-dim (or full-dim h-dim) rope-proportion)]
         (reshape (transpose res-4d [0 2 1 3]) [batch seq-len q-dim]))
       (let [[batch num-heads seq-len h-dim] shape
             r-dim (or rotary-dim h-dim)
             f-dim (or full-dim h-dim)
             rope-prop (or rope-proportion (if (< r-dim h-dim) (/ (double r-dim) (double h-dim)) 1.0))
             half-dim (quot f-dim 2)
             x1 (slice x [0 0 0 0] [batch num-heads seq-len half-dim] [1 1 1 1])
             x2 (slice x [0 0 0 half-dim] [batch num-heads seq-len f-dim] [1 1 1 1])
             neg-x2 (- x2)
             x-rot (concatenate [neg-x2 x1] -1)
             [cos-t sin-t] (generate-rope-freq-tensors seq-len f-dim t-base pos-ids f-dim rope-prop)
             res (+ (* x cos-t) (* x-rot sin-t))]
         (if (= dtype :f32) res (convert res dtype)))))))

(defn apply-rope
  "Applies Rotary Position Embeddings (RoPE) to query or key tensor `x` (or `[q k]`) based on sequence position indices `pos-ids`."
  ([x pos-ids]
   (apply-rope x nil pos-ids nil 10000.0 nil nil 1.0))
  ([x y z]
   (if (number? z)
     (apply-rope x nil y z 10000.0 nil nil 1.0)
     (apply-rope x y z nil 10000.0 nil nil 1.0)))
  ([x y z head-dim]
   (apply-rope x y z head-dim 10000.0 nil nil 1.0))
  ([x y z head-dim theta-base]
   (apply-rope x y z head-dim theta-base nil nil 1.0))
  ([x y z head-dim theta-base rotary-dim]
   (apply-rope x y z head-dim theta-base rotary-dim nil 1.0))
  ([x y z head-dim theta-base rotary-dim full-dim]
   (apply-rope x y z head-dim theta-base rotary-dim full-dim 1.0))
  ([x y z head-dim theta-base rotary-dim full-dim rope-proportion]
   (if (some? y)
     [(apply-rope x nil z head-dim theta-base rotary-dim full-dim rope-proportion)
      (apply-rope y nil z head-dim theta-base rotary-dim full-dim rope-proportion)]
     (let [h-dim (or head-dim
                     (let [shape (second (:type x))]
                       (if (= (count shape) 4) (nth shape 3) 64)))]
       (apply-rope-single x z h-dim theta-base rotary-dim (or full-dim h-dim) rope-proportion)))))

(defn- update-single-cache [cache new-val pos]
  (if (nil? cache)
    [new-val new-val]
    (let [c-type (if (clj-xla.tensor/tracer? cache) (:type cache) cache)
          [_ shape cache-dtype] c-type
          rank (count shape)
          seq-dim (clojure.core/- rank 2)
          max-len (nth shape seq-dim)
          nv-type (if (clj-xla.tensor/tracer? new-val) (:type new-val) new-val)
          [_ new-shape nv-dtype] nv-type
          len (nth new-shape seq-dim)
          pos-val (if (clj-xla.tensor/tracer? pos) pos (extract-pos-int pos))
          starts (assoc (vec (repeat rank 0)) seq-dim pos-val)
          new-val-typed (if (= cache-dtype nv-dtype) new-val (convert new-val cache-dtype))
          updated (dynamic-update-slice cache new-val-typed starts)
          active (if (clj-xla.tensor/tracer? pos)
                   updated
                   (let [end-pos (clojure.core/+ pos-val len)
                         starts-1 (assoc (vec (repeat rank 0)) seq-dim 0)
                         active-limits (assoc (vec shape) seq-dim end-pos)
                         strides (vec (repeat rank 1))]
                     (if (= end-pos max-len)
                       updated
                       (slice updated starts-1 active-limits strides))))]
      [updated active])))

(defn update-kv-cache
  "Updates pre-allocated K/V tensors at position `pos` with `new-k` and `new-v`.
   Returns `[updated-kv-pair active-kv-pair]`."
  ([past-kv new-k new-v pos]
   (if (vector? past-kv)
     (let [[k-cache v-cache] past-kv
           [updated-k active-k] (update-single-cache k-cache new-k pos)
           [updated-v active-v] (update-single-cache v-cache new-v pos)]
       [[updated-k updated-v] [active-k active-v]])
     (let [[k-cache v-cache] past-kv
           [updated-k active-k] (update-single-cache k-cache new-k pos)
           [updated-v active-v] (update-single-cache v-cache new-v pos)]
       [[updated-k updated-v] [active-k active-v]])))
  ([k-cache v-cache new-k new-v pos]
   (update-kv-cache [k-cache v-cache] new-k new-v pos)))

(defn generate-causal-mask
  ([seq-len] (generate-causal-mask seq-len seq-len 0))
  ([q-len kv-len pos]
   (if (clj-xla.tensor/tracer? pos)
     (let [indices-vec (vec (for [j (range kv-len)] (double j)))
           indices-const (clj-xla.tensor/emit-constant! [[[[indices-vec]]]] [:tensor [1 1 1 kv-len] :f32])
           zero-const (clj-xla.tensor/emit-constant! 0.0 [:tensor [] :f32])
           mask-scale (clj-xla.tensor/emit-constant! -10000.0 [:tensor [] :f32])
           pos-f32 (convert pos :f32)
           pos-4d (reshape pos-f32 [1 1 1 1])
           diff (- indices-const pos-4d)
           relu-diff (maximum diff zero-const)
           causal-mask (* relu-diff mask-scale)]
       causal-mask)
     (let [p (extract-pos-int pos)]
       (clj-xla.tensor/emit-constant!
        [[(vec (for [i (range q-len)]
                 (vec (for [j (range kv-len)]
                        (if (clojure.core/<= j (clojure.core/+ p i)) 0.0 -10000.0)))))]]
        [:tensor [1 1 q-len kv-len] :f32])))))

(defn causal-self-attention
  "Multi-head Causal Self-Attention block for GPT-2."
  ([x c-attn-w c-attn-b c-proj-w c-proj-b num-heads]
   (causal-self-attention x c-attn-w c-attn-b c-proj-w c-proj-b num-heads nil nil))
  ([x c-attn-w c-attn-b c-proj-w c-proj-b num-heads past-kv pos]
   (let [[_ [batch seq-len _] _] (:type x)
         num-heads (or num-heads 12)
         ;; 1. Linear QKV Projection
         qkv (linear x c-attn-w c-attn-b)

         ;; 2. Slice into Q, K, V
         [_ [_ _ qkv-dim] _] (:type qkv)
         embed-dim (quot qkv-dim 3)
         head-dim (quot embed-dim num-heads)
         q (slice qkv [0 0 0] [batch seq-len embed-dim] [1 1 1])
         k (slice qkv [0 0 embed-dim] [batch seq-len (clojure.core/* 2 embed-dim)] [1 1 1])
         v (slice qkv [0 0 (clojure.core/* 2 embed-dim)] [batch seq-len qkv-dim] [1 1 1])

         ;; 3. Reshape and Transpose to Multi-Head Shape
         q-heads (transpose (reshape q [batch seq-len num-heads head-dim]) [0 2 1 3])
         k-heads (transpose (reshape k [batch seq-len num-heads head-dim]) [0 2 1 3])
         v-heads (transpose (reshape v [batch seq-len num-heads head-dim]) [0 2 1 3])

         ;; 3b. Update KV Cache if past-kv present
         [updated-kv [k-active v-active]]
         (if (some? past-kv)
           (update-kv-cache past-kv k-heads v-heads (or pos 0))
           [nil [k-heads v-heads]])

         [_ [_ _ kv-len _] _] (:type k-active)
         scale (/ 1.0 (Math/sqrt (double head-dim)))

         ;; 4. QK^T Batched MatMul
         raw-scores (dot-general q-heads k-active
                                 {:batch_dims {:lhs [0 1] :rhs [0 1]}
                                  :contracting_dims {:lhs [3] :rhs [3]}})
         scaled-scores (* raw-scores scale)

         ;; 5. Causal Mask and Softmax
         causal-mask (generate-causal-mask seq-len kv-len (or pos 0))
         masked-scores (+ scaled-scores causal-mask)
         max-s (reduce-max masked-scores :axes [-1] :keep-dims true)
         exp-s (exp (- masked-scores max-s))
         sum-s (reduce-sum exp-s :axes [-1] :keep-dims true)
         probs (/ exp-s sum-s)

         ;; 6. Attention Weights @ V
         context (dot-general probs v-active
                              {:batch_dims {:lhs [0 1] :rhs [0 1]}
                               :contracting_dims {:lhs [3] :rhs [2]}})

         ;; 7. Reshape Back
         merged (reshape (transpose context [0 2 1 3]) [batch seq-len embed-dim])

         ;; 8. Output Linear Projection
         attn-out (linear merged c-proj-w c-proj-b)]
     (if (some? past-kv)
       [attn-out updated-kv]
       attn-out))))

(defn gqa-causal-attention
  "Grouped-Query Causal Self-Attention with causal mask, Softmax, and optional attention logit soft-capping."
  ([q k v o-w num-heads num-kv-heads]
   (gqa-causal-attention q k v o-w num-heads num-kv-heads nil nil nil {}))
  ([q k v o-w num-heads num-kv-heads attn-softcap]
   (gqa-causal-attention q k v o-w num-heads num-kv-heads attn-softcap nil nil {}))
  ([q k v o-w num-heads num-kv-heads attn-softcap past-kv pos]
   (gqa-causal-attention q k v o-w num-heads num-kv-heads attn-softcap past-kv pos {}))
  ([q k v o-w num-heads num-kv-heads attn-softcap past-kv pos opts]
   (let [q-type (:type q)
         k-type (:type k)
         is-q-4d (= (count (second q-type)) 4)
         is-k-4d (= (count (second k-type)) 4)
         batch (first (second q-type))
         q-len (if is-q-4d (nth (second q-type) 2) (nth (second q-type) 1))
         q-dim (if is-q-4d (clojure.core/* (nth (second q-type) 1) (nth (second q-type) 3)) (nth (second q-type) 2))
         num-heads (or num-heads 8)
         num-kv-heads (or num-kv-heads 4)
         head-dim (quot q-dim num-heads)

         k-dim (if is-k-4d (clojure.core/* (nth (second k-type) 1) (nth (second k-type) 3)) (nth (second k-type) 2))
         actual-nkv (if (pos? head-dim) (quot k-dim head-dim) num-kv-heads)
         k-4d (if is-k-4d k (transpose (reshape k [batch q-len actual-nkv head-dim]) [0 2 1 3]))
         v-4d (if is-k-4d v (transpose (reshape v [batch q-len actual-nkv head-dim]) [0 2 1 3]))

         [updated-kv [k-act v-act]]
         (if (some? past-kv)
           (update-kv-cache past-kv k-4d v-4d (or pos 0))
           [nil [k-4d v-4d]])

         k-act-type (:type k-act)
         kv-len (nth (second k-act-type) 2)
         group-size (quot num-heads num-kv-heads)
         scale (get opts :scale (/ 1.0 (Math/sqrt (double head-dim))))

         ;; 1. Reshape & Transpose Q
         q-heads (if is-q-4d
                   q
                   (transpose (reshape q [batch q-len num-heads head-dim]) [0 2 1 3]))

         ;; 2. K and V are already 4D
         k-kvh k-act
         v-kvh v-act

         ;; 3. Repeat KV heads to Q heads if group-size > 1
         [k-heads v-heads] (if (= group-size 1)
                             [k-kvh v-kvh]
                             (let [k-rep (broadcast-in-dim (reshape k-kvh [batch num-kv-heads 1 kv-len head-dim])
                                                           [batch num-kv-heads group-size kv-len head-dim]
                                                           [0 1 2 3 4])
                                   v-rep (broadcast-in-dim (reshape v-kvh [batch num-kv-heads 1 kv-len head-dim])
                                                           [batch num-kv-heads group-size kv-len head-dim]
                                                           [0 1 2 3 4])]
                               [(reshape k-rep [batch num-heads kv-len head-dim])
                                (reshape v-rep [batch num-heads kv-len head-dim])]))

         ;; 4. QK^T Batched MatMul
         raw-scores (dot-general q-heads k-heads
                                 {:batch_dims {:lhs [0 1] :rhs [0 1]}
                                  :contracting_dims {:lhs [3] :rhs [3]}})
         scaled-scores (* raw-scores scale)

         ;; 4b. Attention Logit Soft-Capping
         capped-scores (if (and (number? attn-softcap) (pos? attn-softcap))
                         (* attn-softcap (tanh (/ scaled-scores attn-softcap)))
                         scaled-scores)

         ;; 5. Causal Mask and Softmax
         causal-mask (generate-causal-mask q-len kv-len (or pos 0))
         masked-scores (+ capped-scores causal-mask)
         max-s (reduce-max masked-scores :axes [-1] :keep-dims true)
         exp-s (exp (- masked-scores max-s))
         sum-s (reduce-sum exp-s :axes [-1] :keep-dims true)
         probs (/ exp-s sum-s)

         ;; 6. Attention Weights @ V
         context (dot-general probs v-heads
                              {:batch_dims {:lhs [0 1] :rhs [0 1]}
                               :contracting_dims {:lhs [3] :rhs [2]}})

         ;; 7. Reshape Back to [batch, q-len, q-dim]
         merged (reshape (transpose context [0 2 1 3]) [batch q-len q-dim])

         ;; 8. Output Linear Projection
         o-w-t (transpose o-w [1 0])
         attn-out (linear merged o-w-t nil)]
     (if (some? past-kv)
       [attn-out updated-kv]
       attn-out))))

