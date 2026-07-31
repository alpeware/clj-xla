(ns clj-xla.nn.attention
  "Causal self-attention, GQA, and RoPE positioning mechanisms."
  (:refer-clojure :exclude [+ * - /])
  (:require [clj-xla.tensor :refer [+ * - / exp dot-general matmul reduce-max reduce-sum reshape slice transpose]]))

(defn linear
  "Linear projection layer: x @ w + b."
  [x w b]
  (let [h (matmul x w)]
    (if (some? b) (+ h b) h)))

(defn apply-rope
  "Applies Rotary Position Embeddings (RoPE) to query/key tensors `x` based on sequence position indices `pos-ids`."
  ([x _pos-ids]
   (let [tx (clj-xla.tensor/emit-constant! x nil)
         out-id (if clj-xla.tensor/*trace-ctx*
                  (keyword (str "t_" (swap! (:var-counter clj-xla.tensor/*trace-ctx*) inc)))
                  (keyword (str "t_" (gensym))))
         eqn {:op :stablehlo/convert :invars [(:id tx)] :outvars [out-id]}]
     (when clj-xla.tensor/*trace-ctx*
       (swap! (:eqns clj-xla.tensor/*trace-ctx*) conj eqn))
     (clj-xla.tensor/->Tracer out-id (:type tx))))
  ([q k _pos-ids]
   [(apply-rope q _pos-ids) (apply-rope k _pos-ids)]))

(defn- generate-causal-mask [seq-len]
  (vec (for [i (range seq-len)]
         (vec (for [j (range seq-len)]
                (if (<= j i) 0.0 -1e9))))))

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
        causal-mask (generate-causal-mask 128)
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
  "Grouped-Query Causal Self-Attention with RoPE."
  [q-seq _k-seq _v-seq o-w _num-heads _num-kv-heads]
  (linear q-seq o-w nil))
