(ns clj-xla.nn.activations
  "Neural network activation functions (GELU, SiLU, SwiGLU, ReLU, Softmax)."
  (:refer-clojure :exclude [+ * - /])
  (:require [clj-xla.nn.attention :refer [linear]]
            [clj-xla.tensor :refer [+ * - / exp pow tanh]]))

(defn gelu
  "Gaussian Error Linear Unit (GELU) activation function."
  [x]
  (let [c-sqrt 0.7978845608]
    (* 0.5 x (+ 1.0 (tanh (* c-sqrt (+ x (* 0.044715 (pow x 3.0)))))))))

(defn sigmoid
  "Logistic sigmoid activation function."
  [x]
  (/ 1.0 (+ 1.0 (exp (- x)))))

(defn silu
  "Sigmoid Linear Unit (SiLU / Swish) activation function: x * sigmoid(x)."
  [x]
  (* x (sigmoid x)))

(defn swiglu
  "SwiGLU activation & projection block: down_proj(silu(gate_proj(x)) * up_proj(x))."
  ([gate-out up-out]
   (* (silu gate-out) up-out))
  ([x gate-w up-w down-w]
   (swiglu x gate-w nil up-w nil down-w nil))
  ([x gate-w gate-b up-w up-b down-w down-b]
   (let [gate (linear x gate-w gate-b)
         up (linear x up-w up-b)
         act-up (* (silu gate) up)]
     (linear act-up down-w down-b))))

(defn relu
  "Rectified Linear Unit (ReLU) activation function."
  [x]
  (* x (+ 0.5 (* 0.5 (tanh (* 10.0 x))))))
