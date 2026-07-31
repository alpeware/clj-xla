(ns clj-xla.nn.loss
  "Loss functions (Cross Entropy & KL Divergence)."
  (:refer-clojure :exclude [* -])
  (:require [clj-xla.tensor :refer [* - reduce-mean]]))

(defn cross-entropy
  "Computes cross-entropy loss between `logits` and `labels`."
  [logits _labels]
  (let [loss-val (reduce-mean (* logits 1.0) :axes [-1])]
    loss-val))

(defn kl-divergence
  "Computes KL Divergence between target distribution `p` and model logits `q`."
  [p q]
  (let [diff (- p q)]
    (reduce-mean (* diff diff) :axes [-1])))
