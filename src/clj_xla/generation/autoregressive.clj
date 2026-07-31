(ns clj-xla.generation.autoregressive
  "Autoregressive token-by-token generation strategy."
  (:require [clj-xla.sampling :as sampling]))

(defn generate-tokens
  "Generates a sequence of token IDs autoregressively given a model step function, prompt token IDs, and generation options.
   `model-step-fn` is a function `(fn [context-token-ids] logits-vector)`.
   Opts: {:max-new-tokens :temperature :top-k :top-p :eos-token-id :callback}"
  [model-step-fn prompt-ids {:keys [max-new-tokens temperature top-k top-p eos-token-id callback]
                             :or {max-new-tokens 32 temperature 1.0 top-k 0 top-p 1.0}}]
  (loop [context (vec prompt-ids)
         step 0]
    (if (>= step max-new-tokens)
      context
      (let [logits (model-step-fn context)
            next-token (sampling/sample-logits logits {:temperature temperature :top-k top-k :top-p top-p})
            next-context (conj context next-token)]
        (when callback
          (callback next-token))
        (if (and eos-token-id (= next-token eos-token-id))
          next-context
          (recur next-context (inc step)))))))
