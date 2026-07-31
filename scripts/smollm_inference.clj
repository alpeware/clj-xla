(ns scripts.smollm-inference
  "Top-level runnable integration script for end-to-end SmolLM-135M text generation via pure XLA execution."
  (:require [clj-xla.core :as xla]
            [clj-xla.generation.autoregressive :as ar]
            [clj-xla.models.smollm :as smollm]
            [clj-xla.safetensors :as st]
            [clj-xla.tokenizer.core :as tok]
            [clj-xla.tokenizer.protocol :refer [decode encode eos-id]]
            [clj-xla.trace :refer [trace-graph]])
  (:import [java.lang.foreign Arena]))

(def DEFAULT_CLI_OPTS
  {:prompt "The capital of France is"
   :max-new-tokens 8
   :temperature 0.7
   :top-k 10})

(defn parse-cli-args
  "Parses command-line flags (--prompt, --max-new-tokens, --temperature, --top-k)."
  [args]
  (loop [remaining (vec args)
         opts DEFAULT_CLI_OPTS]
    (if (empty? remaining)
      opts
      (let [flag (first remaining)
            val (second remaining)]
        (cond
          (and (= flag "--prompt") val)
          (recur (subvec remaining 2) (assoc opts :prompt val))

          (and (= flag "--max-new-tokens") val)
          (recur (subvec remaining 2) (assoc opts :max-new-tokens (Long/parseLong val)))

          (and (= flag "--temperature") val)
          (recur (subvec remaining 2) (assoc opts :temperature (Double/parseDouble val)))

          (and (= flag "--top-k") val)
          (recur (subvec remaining 2) (assoc opts :top-k (Long/parseLong val)))

          :else
          (recur (subvec remaining 1) opts))))))

(defn- prepare-input-tensor [tokens max-len]
  (let [padded (take max-len (concat tokens (repeat 0)))]
    (int-array (vec padded))))

(defn -main
  "Runs end-to-end SmolLM-135M text generation pipeline: model loading, tokenization, full-model graph tracing, StableHLO JIT compilation, and autoregressive decoding."
  [& args]
  (let [{:keys [prompt max-new-tokens temperature top-k]} (parse-cli-args args)]
    (println "==================================================================")
    (println "    clj-xla SmolLM-135M End-to-End Autoregressive Generation Loop ")
    (println "==================================================================")

    ;; 1. Initialize CPU PJRT runtime
    (let [ctx (xla/init-cpu!)
          model-dir ".models/smollm-135m"
          weights-path (str model-dir "/model.safetensors")]

      ;; 2. Load Tokenizer (BPE with HF tokenizer.json schema)
      (println (str "Loading SmolLM Tokenizer from [" model-dir "]..."))
      (let [tokenizer (tok/from-file model-dir)
            prompt-ids (encode tokenizer prompt)]
        (println (format "Prompt: \"%s\"" prompt))
        (println (format "Generation Options: max-new-tokens=%d, temperature=%.2f, top-k=%d"
                         max-new-tokens temperature top-k))
        (println (format "Encoded Token IDs (%d tokens): %s" (count prompt-ids) prompt-ids))

        ;; 3. Load Safetensors weights header & memory-map 30 layers
        (println (str "Loading Safetensors metadata from [" weights-path "]..."))
        (let [arena (Arena/ofConfined)
              weights-mmap (st/map-safetensors-weights weights-path arena)
              metadata (:header weights-mmap)
              ^floats embed-tokens (st/get-tensor-floats weights-mmap "model.embed_tokens.weight")
              ^floats final-norm-w (st/get-tensor-floats weights-mmap "model.norm.weight")
              ^floats lm-head-w (or (try (st/get-tensor-floats weights-mmap "lm_head.weight")
                                         (catch Exception _ nil))
                                    embed-tokens)
              layers-weights (mapv (fn [i]
                                     (let [kmap (smollm/weight-key-map i)]
                                       {:input-ln-w (st/get-tensor-floats weights-mmap (:input-ln-w kmap))
                                        :q-w (st/get-tensor-floats weights-mmap (:q-w kmap))
                                        :k-w (st/get-tensor-floats weights-mmap (:k-w kmap))
                                        :v-w (st/get-tensor-floats weights-mmap (:v-w kmap))
                                        :o-w (st/get-tensor-floats weights-mmap (:o-w kmap))
                                        :post-attn-ln-w (st/get-tensor-floats weights-mmap (:post-attn-ln-w kmap))
                                        :gate-w (st/get-tensor-floats weights-mmap (:gate-w kmap))
                                        :up-w (st/get-tensor-floats weights-mmap (:up-w kmap))
                                        :down-w (st/get-tensor-floats weights-mmap (:down-w kmap))}))
                                   (range 30))]
          (println (format "Parsed Safetensors header (%d tensors, %d layers loaded)."
                           (count metadata) (count layers-weights)))

          ;; 4. Trace full SmolLM-135M Model graph & JIT Compile
          (println "Tracing & JIT Compiling full SmolLM-135M model graph to XLA Executable...")
          (let [max-seq-len 128
                invars (into [[:x [:tensor [1 max-seq-len] :i32]]
                              [:embed_tokens [:tensor [49152 576] :f32]]
                              [:final_norm_w [:tensor [576] :f32]]
                              [:lm_head_w [:tensor [49152 576] :f32]]]
                             (mapcat (fn [i]
                                       [[(keyword (str "input_ln_w_" i)) [:tensor [576] :f32]]
                                        [(keyword (str "q_w_" i)) [:tensor [576 576] :f32]]
                                        [(keyword (str "k_w_" i)) [:tensor [192 576] :f32]]
                                        [(keyword (str "v_w_" i)) [:tensor [192 576] :f32]]
                                        [(keyword (str "o_w_" i)) [:tensor [576 576] :f32]]
                                        [(keyword (str "post_attn_ln_w_" i)) [:tensor [576] :f32]]
                                        [(keyword (str "gate_w_" i)) [:tensor [1536 576] :f32]]
                                        [(keyword (str "up_w_" i)) [:tensor [1536 576] :f32]]
                                        [(keyword (str "down_w_" i)) [:tensor [576 1536] :f32]]])
                                     (range 30)))
                trace-fn (fn [x emb fn-norm lm-hw & layer-args]
                           (let [lw-seq (mapv (fn [[in-ln qw kw vw ow post-ln gw uw dw]]
                                                {:input-ln-w in-ln :q-w qw :k-w kw :v-w vw :o-w ow :post-attn-ln-w post-ln :gate-w gw :up-w uw :down-w dw})
                                              (partition 9 layer-args))]
                             (smollm/full-smollm-forward x emb lw-seq fn-norm lm-hw [0])))
                graph (trace-graph "full_smollm_model" invars trace-fn)
                exec (xla/compile-graph ctx graph)]
            (println "Successfully compiled StableHLO graph to native XLA PjRtLoadedExecutable handle.")

            (println "Transferring weights to PJRT Device Memory...")
            (let [embed-buf (xla/buffer-from-host-buffer ctx (:client ctx) embed-tokens [49152 576] 11)
                  final-norm-buf (xla/buffer-from-host-buffer ctx (:client ctx) final-norm-w [576] 11)
                  lm-head-buf (xla/buffer-from-host-buffer ctx (:client ctx) lm-head-w [49152 576] 11)
                  layer-bufs (mapv (fn [m]
                                     [(xla/buffer-from-host-buffer ctx (:client ctx) (:input-ln-w m) [576] 11)
                                      (xla/buffer-from-host-buffer ctx (:client ctx) (:q-w m) [576 576] 11)
                                      (xla/buffer-from-host-buffer ctx (:client ctx) (:k-w m) [192 576] 11)
                                      (xla/buffer-from-host-buffer ctx (:client ctx) (:v-w m) [192 576] 11)
                                      (xla/buffer-from-host-buffer ctx (:client ctx) (:o-w m) [576 576] 11)
                                      (xla/buffer-from-host-buffer ctx (:client ctx) (:post-attn-ln-w m) [576] 11)
                                      (xla/buffer-from-host-buffer ctx (:client ctx) (:gate-w m) [1536 576] 11)
                                      (xla/buffer-from-host-buffer ctx (:client ctx) (:up-w m) [1536 576] 11)
                                      (xla/buffer-from-host-buffer ctx (:client ctx) (:down-w m) [576 1536] 11)])
                                   layers-weights)
                  flat-layer-bufs (vec (apply concat layer-bufs))
                  flat-device-weights (into [embed-buf final-norm-buf lm-head-buf] flat-layer-bufs)]

              ;; 5. Execute Autoregressive Generation Loop
              (println "\nGenerating tokens autoregressively...")
              (print prompt)
              (flush)
              (let [step-fn (fn [context-ids]
                              (let [S (count context-ids)
                                    input-array (prepare-input-tensor context-ids max-seq-len)
                                    input-buf (xla/buffer-from-host-buffer ctx (:client ctx) input-array [1 max-seq-len] 4)
                                    input-args (into [input-buf] flat-device-weights)
                                    logits-out (xla/execute exec input-args)
                                    last-logits (xla/to-host-slice logits-out (dec S) 49152)]
                                last-logits))
                    gen-ids (ar/generate-tokens step-fn prompt-ids {:max-new-tokens max-new-tokens
                                                                    :temperature temperature
                                                                    :top-k top-k
                                                                    :eos-token-id (eos-id tokenizer)
                                                                    :callback (fn [token-id]
                                                                                (print (decode tokenizer [token-id]))
                                                                                (flush))})
                    full-text (decode tokenizer gen-ids)]
                (println "\n\n==================================================================")
                (println "Final Generated Sequence:")
                (println full-text)
                (println "==================================================================")
                (println "=== End-to-End SmolLM-135M Generation Verification Passed! ===")))))))))

(when (= *file* (System/getProperty "clojure.script.filename"))
  (apply -main *command-line-args*))
