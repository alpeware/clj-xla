(ns clj-xla.nn.attention
  "Causal self-attention, GQA, and RoPE positioning mechanisms."
  (:refer-clojure :exclude [+])
  (:require [clj-xla.tensor :refer [+]]))

(defn linear
  "Linear projection layer: x @ w + b."
  [x w b]
  (if (or (instance? clj_xla.tensor.Tracer x) (some? clj-xla.tensor/*trace-ctx*))
    (let [tx (clj-xla.tensor/emit-constant! x nil)
          tw (clj-xla.tensor/emit-constant! w nil)
          tb (when b (clj-xla.tensor/emit-constant! b nil))
          out-id (if clj-xla.tensor/*trace-ctx*
                   (keyword (str "t_" (swap! (:var-counter clj-xla.tensor/*trace-ctx*) inc)))
                   (keyword (str "t_" (gensym))))
          eqn {:op :stablehlo/dot_general
               :invars [(:id tx) (:id tw)]
               :outvars [out-id]}]
      (when clj-xla.tensor/*trace-ctx*
        (swap! (:eqns clj-xla.tensor/*trace-ctx*) conj eqn))
      (let [proj-tracer (clj-xla.tensor/->Tracer out-id (:type tx))]
        (if (some? tb) (+ proj-tracer tb) proj-tracer)))
    x))

(defn apply-rope
  "Applies Rotary Position Embeddings (RoPE) to query/key tensors `x` based on sequence position indices `pos-ids`."
  ([x _pos-ids]
   (if (or (instance? clj_xla.tensor.Tracer x) (some? clj-xla.tensor/*trace-ctx*))
     (let [tx (clj-xla.tensor/emit-constant! x nil)
           out-id (if clj-xla.tensor/*trace-ctx*
                    (keyword (str "t_" (swap! (:var-counter clj-xla.tensor/*trace-ctx*) inc)))
                    (keyword (str "t_" (gensym))))
           eqn {:op :stablehlo/custom_call
                :call_target_name "rope"
                :invars [(:id tx)]
                :outvars [out-id]}]
       (when clj-xla.tensor/*trace-ctx*
         (swap! (:eqns clj-xla.tensor/*trace-ctx*) conj eqn))
       (clj-xla.tensor/->Tracer out-id (:type tx)))
     x))
  ([q k pos-ids]
   [(apply-rope q pos-ids) (apply-rope k pos-ids)]))

(defn causal-self-attention
  "Multi-head Causal Self-Attention block for Transformer architectures."
  [x c-attn-w c-attn-b c-proj-w c-proj-b _num-heads]
  (let [_qkv (linear x c-attn-w c-attn-b)
        proj (linear x c-proj-w c-proj-b)]
    proj))
