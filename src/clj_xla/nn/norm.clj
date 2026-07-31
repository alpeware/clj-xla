(ns clj-xla.nn.norm
  "Neural network normalization layers (Layer Normalization & RMS Normalization)."
  (:refer-clojure :exclude [+ * - /])
  (:require [clj-xla.tensor :refer [+ * - / pow reduce-mean sqrt]]))

(defn layer-norm
  "Layer Normalization layer: (x - mu) / sqrt(var + eps) * gamma + beta."
  ([x gamma beta]
   (layer-norm x gamma beta 1e-5))
  ([x gamma beta eps]
   (let [mu (reduce-mean x :axes [-1] :keep-dims true)
         diff (- x mu)
         var-val (reduce-mean (pow diff 2.0) :axes [-1] :keep-dims true)
         x-hat (/ diff (sqrt (+ var-val eps)))]
     (+ (* x-hat gamma) beta))))

(defn rms-norm
  "Root Mean Square Normalization (RMSNorm) layer: x / sqrt(mean(x^2) + eps) * weight."
  ([x weight]
   (rms-norm x weight 1e-6))
  ([x weight eps]
   (let [ms (reduce-mean (pow x 2.0) :axes [-1] :keep-dims true)
         x-hat (/ x (sqrt (+ ms eps)))]
     (* x-hat weight))))
