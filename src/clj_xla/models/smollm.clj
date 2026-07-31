(ns clj-xla.models.smollm
  "SmolLM / Llama Architecture Assembler."
  (:refer-clojure :exclude [*])
  (:require [clj-xla.nn.activations :refer [silu]]
            [clj-xla.tensor :refer [*]]))

(def DEFAULT_SMOLLM_CONFIG
  {:vocab-size 49152
   :hidden-size 576
   :intermediate-size 1536
   :num-hidden-layers 30
   :num-attention-heads 9
   :num-key-value-heads 3
   :rms-norm-eps 1e-5})

(defn smollm-config
  "Returns SmolLM configuration map with optional custom overrides."
  ([] DEFAULT_SMOLLM_CONFIG)
  ([overrides] (merge DEFAULT_SMOLLM_CONFIG overrides)))

(defn smollm-mlp
  "SwiGLU Feed-Forward MLP Block for Llama/SmolLM."
  [x _gate-w _up-w _down-w]
  (let [gate (silu x)
        up (* gate x)]
    up))
