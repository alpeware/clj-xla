(ns clj-xla.example.kernels
  "High-level neural network kernels implemented in pure Clojure."
  (:refer-clojure :exclude [+ * - /])
  (:require [clj-xla.tensor :refer [+ * - / pow reduce-mean sqrt tanh]]))

(defn gelu
  "Gaussian Error Linear Unit (GELU) activation function."
  [x]
  (let [c-sqrt 0.7978845608]
    (* 0.5 x (+ 1.0 (tanh (* c-sqrt (+ x (* 0.044715 (pow x 3.0)))))))))

(defn layer-norm
  "Layer Normalization kernel: (x - mu) / sqrt(var + eps) * gamma + beta."
  [x gamma beta eps]
  (let [mu (reduce-mean x :axes [-1] :keep-dims true)
        diff (- x mu)
        var-val (reduce-mean (pow diff 2.0) :axes [-1] :keep-dims true)
        x-hat (/ diff (sqrt (+ var-val eps)))]
    (+ (* x-hat gamma) beta)))
