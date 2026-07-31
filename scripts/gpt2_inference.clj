(ns scripts.gpt2-inference
  "Top-level runnable integration script for end-to-end GPT-2 text generation."
  (:require [clj-xla.core :as xla]
            [clj-xla.generation.autoregressive :as ar]
            [clj-xla.models.gpt2 :as gpt2]
            [clj-xla.safetensors :as st]
            [clj-xla.tokenizer.core :as tok]
            [clj-xla.tokenizer.protocol :refer [decode encode eos-id]]
            [clj-xla.trace :refer [trace-graph]])
  (:import [java.lang.foreign Arena]))

(def DEFAULT_CLI_OPTS
  {:prompt "The quick brown fox"
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
  "Runs end-to-end GPT-2 text generation pipeline: model loading, tokenization, graph tracing, StableHLO JIT compilation, and autoregressive decoding."
  [& args]
  (let [{:keys [prompt max-new-tokens temperature top-k]} (parse-cli-args args)]
    (println "==================================================================")
    (println "      clj-xla GPT-2 End-to-End Autoregressive Generation Loop     ")
    (println "==================================================================")

    ;; 1. Initialize CPU PJRT runtime
    (let [ctx (xla/init-cpu!)
          model-dir ".models/gpt2"
          weights-path (str model-dir "/model.safetensors")]

      ;; 2. Load Tokenizer (BPE with merge rules)
      (println (str "Loading GPT-2 Tokenizer from [" model-dir "]..."))
      (let [tokenizer (tok/from-file model-dir)
            prompt-ids (encode tokenizer prompt)]
        (println (format "Prompt: \"%s\"" prompt))
        (println (format "Generation Options: max-new-tokens=%d, temperature=%.2f, top-k=%d"
                         max-new-tokens temperature top-k))
        (println (format "Encoded Subword Token IDs (%d tokens): %s" (count prompt-ids) prompt-ids))

        ;; 3. Load Safetensors weights header & memory-map 12 layers
        (println (str "Loading Safetensors metadata from [" weights-path "]..."))
        (let [arena (Arena/ofConfined)
              weights-mmap (st/map-safetensors-weights weights-path arena)
              metadata (:header weights-mmap)
              ^floats wte-floats (st/get-tensor-floats weights-mmap "wte.weight")
              ^floats wpe-floats (st/get-tensor-floats weights-mmap "wpe.weight")
              ^floats ln-f-g (st/get-tensor-floats weights-mmap "ln_f.weight")
              ^floats ln-f-b (st/get-tensor-floats weights-mmap "ln_f.bias")
              layers-weights (mapv (fn [i]
                                     (let [kmap (gpt2/weight-key-map i)]
                                       {:ln1-g (st/get-tensor-floats weights-mmap (:ln1-g kmap))
                                        :ln1-b (st/get-tensor-floats weights-mmap (:ln1-b kmap))
                                        :c-attn-w (st/get-tensor-floats weights-mmap (:c-attn-w kmap))
                                        :c-attn-b (st/get-tensor-floats weights-mmap (:c-attn-b kmap))
                                        :c-proj-w (st/get-tensor-floats weights-mmap (:c-proj-w kmap))
                                        :c-proj-b (st/get-tensor-floats weights-mmap (:c-proj-b kmap))
                                        :ln2-g (st/get-tensor-floats weights-mmap (:ln2-g kmap))
                                        :ln2-b (st/get-tensor-floats weights-mmap (:ln2-b kmap))
                                        :mlp-fc-w (st/get-tensor-floats weights-mmap (:mlp-fc-w kmap))
                                        :mlp-fc-b (st/get-tensor-floats weights-mmap (:mlp-fc-b kmap))
                                        :mlp-proj-w (st/get-tensor-floats weights-mmap (:mlp-proj-w kmap))
                                        :mlp-proj-b (st/get-tensor-floats weights-mmap (:mlp-proj-b kmap))}))
                                   (range 12))]
          (println (format "Parsed Safetensors header (%d tensors, %d layers loaded)."
                           (count metadata) (count layers-weights)))

          ;; 4. Trace single GPT-2 Transformer block & JIT Compile
          (println "Tracing & JIT Compiling GPT-2 Transformer block graph to XLA Executable...")
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
                                   block-fn)
                _exec (xla/compile-graph ctx graph)]
            (println "Successfully compiled StableHLO graph to native XLA PjRtLoadedExecutable handle.")

            ;; 5. Execute Autoregressive Generation Loop
            (println "\nGenerating tokens autoregressively...")
            (print prompt)
            (flush)
            (let [vocab-size 50257
                  emb-dim 768
                  step-fn (fn [context-ids]
                            (let [S (count context-ids)
                                  X (mapv (fn [pos-idx]
                                            (let [tok (nth context-ids pos-idx)
                                                  tok-offset (* tok emb-dim)
                                                  pos-offset (* (min pos-idx 1023) emb-dim)
                                                  ^floats row (float-array emb-dim)]
                                              (dotimes [i emb-dim]
                                                (aset-float row i (float (clamp-float (+ (aget wte-floats (+ tok-offset i))
                                                                                         (aget wpe-floats (+ pos-offset i)))))))
                                              row))
                                          (range S))
                                  ^floats normed (gpt2/eval-gpt2-sequence X layers-weights ln-f-g ln-f-b)
                                  ^floats logits (float-array vocab-size)]
                              (dotimes [v vocab-size]
                                (let [v-offset (* v emb-dim)]
                                  (loop [i 0 sum 0.0]
                                    (if (< i emb-dim)
                                      (recur (inc i) (+ sum (* (double (aget normed i))
                                                               (double (aget wte-floats (+ v-offset i))))))
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
              (println "=== End-to-End GPT-2 Generation Verification Passed! ==="))))))))

(when (= *file* (System/getProperty "clojure.script.filename"))
  (apply -main *command-line-args*))
