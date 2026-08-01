(ns clj-xla.nn.attention
  "Causal self-attention, GQA, and RoPE positioning mechanisms."
  (:refer-clojure :exclude [+ * - /])
  (:require [clj-xla.tensor :refer [+ * - / broadcast-in-dim concatenate exp dot-general matmul reduce-max reduce-sum reshape slice tanh transpose]]))

(defn linear
  "Linear projection layer: x @ w + b."
  [x w b]
  (let [h (matmul x w)]
    (if (some? b) (+ h b) h)))

(defn- generate-rope-freq-tensors [seq-len head-dim theta-base]
  (let [half-dim (quot head-dim 2)
        freqs (vec (for [i (range half-dim)]
                     (Math/pow theta-base (clojure.core// (clojure.core/* -2.0 i) (double head-dim)))))
        cos-rows (vec (for [pos (range seq-len)]
                        (let [half (vec (for [i (range half-dim)]
                                          (Math/cos (clojure.core/* (double pos) (nth freqs i)))))]
                          (vec (clojure.core/concat half half)))))
        sin-rows (vec (for [pos (range seq-len)]
                        (let [half (vec (for [i (range half-dim)]
                                          (Math/sin (clojure.core/* (double pos) (nth freqs i)))))]
                          (vec (clojure.core/concat half half)))))]
    [(clj-xla.tensor/emit-constant! [[cos-rows]] [:tensor [1 1 seq-len head-dim] :f32])
     (clj-xla.tensor/emit-constant! [[sin-rows]] [:tensor [1 1 seq-len head-dim] :f32])]))

(defn apply-rope
  "Applies Rotary Position Embeddings (RoPE) to query or key tensor `x` (or `[q k]`) based on sequence position indices `pos-ids`."
  ([x pos-ids]
   (if (vector? x)
     (mapv #(apply-rope % pos-ids) x)
     (let [[_ [_batch _num-heads _seq-len head-dim] _] (:type x)]
       (apply-rope x pos-ids head-dim))))
  ([x y z]
   (if (number? z)
     (let [head-dim z
           [_ [batch num-heads seq-len _] _] (:type x)
           half-dim (quot head-dim 2)
           x1 (slice x [0 0 0 0] [batch num-heads seq-len half-dim] [1 1 1 1])
           x2 (slice x [0 0 0 half-dim] [batch num-heads seq-len head-dim] [1 1 1 1])
           neg-x2 (- x2)
           x-rot (concatenate [neg-x2 x1] -1)
           [cos-t sin-t] (generate-rope-freq-tensors seq-len head-dim 10000.0)]
       (+ (* x cos-t) (* x-rot sin-t)))
     [(apply-rope x z) (apply-rope y z)])))

(defn- generate-causal-mask [seq-len]
  (vec (for [i (range seq-len)]
         (vec (for [j (range seq-len)]
                (if (<= j i) 0.0 -10000.0))))))

(defn causal-self-attention
  "Multi-head Causal Self-Attention block for GPT-2."
  [x c-attn-w c-attn-b c-proj-w c-proj-b _num-heads]
  (let [;; 1. Linear QKV Projection: [1, 128, 768] -> [1, 128, 2304]
        qkv (linear x c-attn-w c-attn-b)

        ;; 2. Slice into Q, K, V: each [1, 128, 768]
        q (slice qkv [0 0 0] [1 128 768] [1 1 1])
        k (slice qkv [0 0 768] [1 128 1536] [1 1 1])
        v (slice qkv [0 0 1536] [1 128 2304] [1 1 1])

        ;; 3. Reshape and Transpose to Multi-Head Shape: [1, 12, 128, 64]
        q-heads (transpose (reshape q [1 128 12 64]) [0 2 1 3])
        k-heads (transpose (reshape k [1 128 12 64]) [0 2 1 3])
        v-heads (transpose (reshape v [1 128 12 64]) [0 2 1 3])

        ;; 4. QK^T Batched MatMul: [1, 12, 128, 64] x [1, 12, 128, 64] -> [1, 12, 128, 128]
        raw-scores (dot-general q-heads k-heads
                                {:batch_dims {:lhs [0 1] :rhs [0 1]}
                                 :contracting_dims {:lhs [3] :rhs [3]}})
        scaled-scores (* raw-scores 0.125) ;; 1 / sqrt(64)

        ;; 5. Add Causal Mask and Softmax across last dimension
        causal-mask (clj-xla.tensor/emit-constant! [[(generate-causal-mask 128)]] [:tensor [1 1 128 128] :f32])
        masked-scores (+ scaled-scores causal-mask)
        max-s (reduce-max masked-scores :axes [-1] :keep-dims true)
        exp-s (exp (- masked-scores max-s))
        sum-s (reduce-sum exp-s :axes [-1] :keep-dims true)
        probs (/ exp-s sum-s)

        ;; 6. Attention Weights @ V: [1, 12, 128, 128] x [1, 12, 128, 64] -> [1, 12, 128, 64]
        context (dot-general probs v-heads
                             {:batch_dims {:lhs [0 1] :rhs [0 1]}
                              :contracting_dims {:lhs [3] :rhs [2]}})

        ;; 7. Reshape Back to [1, 128, 768]
        merged (reshape (transpose context [0 2 1 3]) [1 128 768])]

    ;; 8. Output Linear Projection: [1, 128, 768] -> [1, 128, 768]
    (linear merged c-proj-w c-proj-b)))

(defn gqa-causal-attention
  "Grouped-Query Causal Self-Attention with causal mask, Softmax, and optional attention logit soft-capping."
  ([q k v o-w num-heads num-kv-heads]
   (gqa-causal-attention q k v o-w num-heads num-kv-heads 50.0))
  ([q k v o-w num-heads num-kv-heads attn-softcap]
   (let [[_ [batch seq-len q-dim] _] (:type q)
         num-heads (or num-heads 8)
         num-kv-heads (or num-kv-heads 4)
         head-dim (quot q-dim num-heads)
         group-size (quot num-heads num-kv-heads)
         scale (/ 1.0 (Math/sqrt (double head-dim)))

         ;; 1. Reshape & Transpose Q: [batch, seq-len, q-dim] -> [batch, num-heads, seq-len, head-dim]
         q-heads (transpose (reshape q [batch seq-len num-heads head-dim]) [0 2 1 3])

         ;; 2. Reshape & Transpose K, V: [batch, seq-len, kv-dim] -> [batch, num-kv-heads, seq-len, head-dim]
         k-kvh (transpose (reshape k [batch seq-len num-kv-heads head-dim]) [0 2 1 3])
         v-kvh (transpose (reshape v [batch seq-len num-kv-heads head-dim]) [0 2 1 3])

         ;; 3. Repeat KV heads to Q heads if group-size > 1
         [k-heads v-heads] (if (= group-size 1)
                             [k-kvh v-kvh]
                             (let [k-rep (broadcast-in-dim (reshape k-kvh [batch num-kv-heads 1 seq-len head-dim])
                                                           [batch num-kv-heads group-size seq-len head-dim]
                                                           [0 1 2 3 4])
                                   v-rep (broadcast-in-dim (reshape v-kvh [batch num-kv-heads 1 seq-len head-dim])
                                                           [batch num-kv-heads group-size seq-len head-dim]
                                                           [0 1 2 3 4])]
                               [(reshape k-rep [batch num-heads seq-len head-dim])
                                (reshape v-rep [batch num-heads seq-len head-dim])]))

         ;; 4. QK^T Batched MatMul
         raw-scores (dot-general q-heads k-heads
                                 {:batch_dims {:lhs [0 1] :rhs [0 1]}
                                  :contracting_dims {:lhs [3] :rhs [3]}})
         scaled-scores (* raw-scores scale)

         ;; 4b. Attention Logit Soft-Capping (Gemma 2: 50.0 * tanh(scores / 50.0))
         capped-scores (if (and (number? attn-softcap) (pos? attn-softcap))
                         (* attn-softcap (tanh (/ scaled-scores attn-softcap)))
                         scaled-scores)

         ;; 5. Add Causal Mask and Softmax
         causal-mask (clj-xla.tensor/emit-constant! [[(generate-causal-mask seq-len)]] [:tensor [1 1 seq-len seq-len] :f32])
         masked-scores (+ capped-scores causal-mask)
         max-s (reduce-max masked-scores :axes [-1] :keep-dims true)
         exp-s (exp (- masked-scores max-s))
         sum-s (reduce-sum exp-s :axes [-1] :keep-dims true)
         probs (/ exp-s sum-s)

         ;; 6. Attention Weights @ V
         context (dot-general probs v-heads
                              {:batch_dims {:lhs [0 1] :rhs [0 1]}
                               :contracting_dims {:lhs [3] :rhs [2]}})

         ;; 7. Reshape Back to [batch, seq-len, q-dim]
         merged (reshape (transpose context [0 2 1 3]) [batch seq-len q-dim])

         ;; 8. Output Linear Projection (transposed weight)
         o-w-t (transpose o-w [1 0])]
     (linear merged o-w-t nil))))
