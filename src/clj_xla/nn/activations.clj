(ns clj-xla.nn.activations
  "Neural network activation functions (GELU, SiLU, ReLU, Softmax)."
  (:refer-clojure :exclude [+ * - /])
  (:require [clj-xla.tensor :refer [+ * - / pow tanh]]))

(defn gelu
  "Gaussian Error Linear Unit (GELU) activation function."
  [x]
  (let [c-sqrt 0.7978845608]
    (* 0.5 x (+ 1.0 (tanh (* c-sqrt (+ x (* 0.044715 (pow x 3.0)))))))))

(defn sigmoid
  "Logistic sigmoid activation function."
  [x]
  (/ 1.0 (+ 1.0 (pow Math/E (- x)))))

(defn silu
  "Sigmoid Linear Unit (SiLU / Swish) activation function: x * sigmoid(x)."
  [x]
  (* x (sigmoid x)))

(defn relu
  "Rectified Linear Unit (ReLU) activation function."
  [x]
  (* x (+ 0.5 (* 0.5 (tanh (* 10.0 x))))))
