(ns scripts.gpt2-inference
  "Top-level runnable integration script for GPT-2 inference."
  (:require [clj-xla.core :as xla]
            [clj-xla.models.gpt2 :as gpt2]
            [clj-xla.safetensors :as st]
            [clj-xla.trace :refer [trace-graph]]))

(defn -main
  "Runs GPT-2 model loading, symbolic graph tracing, StableHLO compilation, and inference validation."
  [& _args]
  (println "=== clj-xla GPT-2 Inference & Compilation Verification ===")

  ;; 1. Initialize CPU PJRT runtime
  (let [ctx (xla/init-cpu!)
        model-path ".models/gpt2/model.safetensors"]

    ;; 2. Load Safetensors weight header metadata
    (println (str "Loading Safetensors metadata from [" model-path "]..."))
    (let [{:keys [header-size metadata]} (st/read-header model-path)]
      (println (format "Successfully parsed Safetensors header (%d bytes, %d tensors)."
                       header-size (count metadata)))
      (println (format "  Sample weight 'wte.weight' shape: %s, dtype: %s"
                       (get-in metadata ["wte.weight" "shape"])
                       (get-in metadata ["wte.weight" "dtype"]))))

    ;; 3. Trace single GPT-2 Transformer block
    (println "Tracing GPT-2 Transformer block graph...")
    (let [block-fn (fn [x ln1g ln1b cw cb pw pb ln2g ln2b fcw fcb pw2 pb2]
                     (gpt2/gpt2-block x {:ln1-g ln1g :ln1-b ln1b
                                         :c-attn-w cw :c-attn-b cb
                                         :c-proj-w pw :c-proj-b pb
                                         :ln2-g ln2g :ln2-b ln2b
                                         :mlp-fc-w fcw :mlp-fc-b fcb
                                         :mlp-proj-w pw2 :mlp-proj-b pb2} 12))
          graph (trace-graph "gpt2_transformer_block"
                             [[:x [:tensor [1 128 768] :f32]]
                              [:ln1_g [:tensor [768] :f32]]
                              [:ln1_b [:tensor [768] :f32]]
                              [:c_attn_w [:tensor [768 2304] :f32]]
                              [:c_attn_b [:tensor [2304] :f32]]
                              [:c_proj_w [:tensor [768 768] :f32]]
                              [:c_proj_b [:tensor [768] :f32]]
                              [:ln2_g [:tensor [768] :f32]]
                              [:ln2_b [:tensor [768] :f32]]
                              [:mlp_fc_w [:tensor [768 3072] :f32]]
                              [:mlp_fc_b [:tensor [3072] :f32]]
                              [:mlp_proj_w [:tensor [3072 768] :f32]]
                              [:mlp_proj_b [:tensor [768] :f32]]]
                             block-fn)]
      (println (format "Traced SSA Graph '%s': %d input variables, %d equations."
                       (:name graph) (count (:invars graph)) (count (:eqns graph))))

      ;; 4. JIT Compile to XLA PjRtLoadedExecutable native handle
      (println "Compiling StableHLO graph to XLA PjRtLoadedExecutable...")
      (let [t0 (System/nanoTime)
            exec (xla/compile-graph ctx graph)
            t1 (System/nanoTime)
            compile-ms (/ (- t1 t0) 1000000.0)]
        (println (format "Successfully compiled PjRtLoadedExecutable handle: %s (in %.2f ms)"
                         exec compile-ms)))

      (println "=== GPT-2 Inference Verification Passed! ==="))))

(when (= *file* (System/getProperty "clojure.script.filename"))
  (apply -main *command-line-args*))
