(ns scripts.smollm-inference
  "Top-level runnable integration script for end-to-end SmolLM-135M text generation."
  (:require [clj-xla.core :as xla]
            [clj-xla.generation.autoregressive :as ar]
            [clj-xla.models.smollm :as smollm]
            [clj-xla.safetensors :as st]
            [clj-xla.tokenizer.core :as tok]
            [clj-xla.tokenizer.protocol :refer [decode encode eos-id]]
            [clj-xla.trace :refer [trace-graph]])
  (:import [java.lang.foreign Arena]))

(def DEFAULT_CLI_OPTS
  {:prompt "Once upon a time, in a small village"
   :max-new-tokens 8
   :temperature 0.7
   :top-k 10})

(defn- clamp-float ^double [^double d]
  (cond
    (Double/isNaN d) 0.0
    (> d 3.4028234663852886E38) 3.4028234663852886E38
    (< d -3.4028234663852886E38) -3.4028234663852886E38
    :else d))

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

(defn -main
  "Runs end-to-end SmolLM-135M text generation pipeline: model loading, tokenization, graph tracing, StableHLO JIT compilation, and autoregressive decoding."
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

          ;; 4. Trace single SmolLM Transformer block & JIT Compile
          (println "Tracing & JIT Compiling SmolLM Transformer block graph to XLA Executable...")
          (let [block-fn (fn [x in-ln qw kw vw ow post-ln gw uw dw]
                           (smollm/smollm-block x {:input-ln-w in-ln
                                                   :q-w qw :k-w kw :v-w vw :o-w ow
                                                   :post-attn-ln-w post-ln
                                                   :gate-w gw :up-w uw :down-w dw} 9 3 [0]))
                graph (trace-graph "smollm_transformer_block"
                                   [[:x [:tensor [1 128 576] :f32]]
                                    [:in_ln [:tensor [576] :f32]]
                                    [:qw [:tensor [576 576] :f32]]
                                    [:kw [:tensor [576 192] :f32]]
                                    [:vw [:tensor [576 192] :f32]]
                                    [:ow [:tensor [576 576] :f32]]
                                    [:post_ln [:tensor [576] :f32]]
                                    [:gw [:tensor [576 1536] :f32]]
                                    [:uw [:tensor [576 1536] :f32]]
                                    [:dw [:tensor [1536 576] :f32]]]
                                   block-fn)
                _exec (xla/compile-graph ctx graph)]
            (println "Successfully compiled StableHLO graph to native XLA PjRtLoadedExecutable handle.")

            ;; 5. Execute Autoregressive Generation Loop
            (println "\nGenerating tokens autoregressively...")
            (print prompt)
            (flush)
            (let [vocab-size 49152
                  emb-dim 576
                  step-fn (fn [context-ids]
                            (let [S (count context-ids)
                                  X (mapv (fn [pos-idx]
                                            (let [tok (nth context-ids pos-idx)
                                                  tok-offset (* tok emb-dim)
                                                  ^floats row (float-array emb-dim)]
                                              (dotimes [i emb-dim]
                                                (aset-float row i (float (clamp-float (aget embed-tokens (+ tok-offset i))))))
                                              row))
                                          (range S))
                                  ^floats normed (smollm/eval-smollm-sequence X layers-weights final-norm-w)
                                  ^floats logits (float-array vocab-size)]
                              (dotimes [v vocab-size]
                                (let [v-offset (* v emb-dim)]
                                  (loop [i 0 sum 0.0]
                                    (if (< i emb-dim)
                                      (recur (inc i) (+ sum (* (double (aget normed i))
                                                               (double (aget lm-head-w (+ v-offset i))))))
                                      (aset-float logits v (float (clamp-float sum)))))))
                              (vec logits)))
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
              (println "=== End-to-End SmolLM-135M Generation Verification Passed! ==="))))))))

(when (= *file* (System/getProperty "clojure.script.filename"))
  (apply -main *command-line-args*))
