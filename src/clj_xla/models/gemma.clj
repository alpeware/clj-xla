(ns clj-xla.models.gemma
  "Gemma 2B Architecture Assembler.")

(def DEFAULT_GEMMA_CONFIG
  {:vocab-size 256000
   :hidden-size 2048
   :intermediate-size 16384
   :num-hidden-layers 18
   :num-attention-heads 8
   :num-key-value-heads 1
   :head-dim 256
   :rms-norm-eps 1e-6})

(defn gemma-config
  "Returns Gemma configuration map with optional custom overrides."
  ([] DEFAULT_GEMMA_CONFIG)
  ([overrides] (merge DEFAULT_GEMMA_CONFIG overrides)))
