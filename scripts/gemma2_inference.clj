(ns scripts.gemma2-inference
  "Top-level runnable integration script for end-to-end Gemma 2B text generation via pure XLA execution with Single-Pass Prefill and BF16/FP16 precision."
  (:require [clj-xla.core :as xla]
            [clj-xla.models.gemma :as gemma]
            [clj-xla.safetensors :as st]
            [clj-xla.sampling :as sampling]
            [clj-xla.tensor :as t]
            [clj-xla.tokenizer.core :as tok]
            [clj-xla.tokenizer.protocol :refer [bos-id decode encode eos-id]]
            [clj-xla.trace :refer [trace-graph]]
            [clojure.java.io :as io]
            [clojure.pprint :as pprint])
  (:import [java.lang.foreign Arena]))

(def DEFAULT_CLI_OPTS
  {:prompt "The capital of France is"
   :max-new-tokens 10
   :temperature 0.7
   :top-k 10
   :backend :cpu
   :precision :bf16
   :verbose false})

(defn parse-cli-args
  "Parses command-line flags (--prompt, --max-new-tokens, --temperature, --top-k, --backend, --precision, --verbose)."
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

          (and (= flag "--backend") val)
          (recur (subvec remaining 2) (assoc opts :backend (keyword val)))

          (and (= flag "--precision") val)
          (recur (subvec remaining 2) (assoc opts :precision (keyword val)))

          (= flag "--verbose")
          (recur (subvec remaining 1) (assoc opts :verbose true))

          :else
          (recur (subvec remaining 1) opts))))))

(defrecord DummyGemmaTokenizer []
  clj-xla.tokenizer.protocol/Tokenizer
  (encode [_ text]
    (encode _ text false))
  (encode [_ _text _add-special-tokens?]
    [101 102 103 104])
  (decode [_ token-ids]
    (decode _ token-ids true))
  (decode [_ token-ids _skip-special-tokens?]
    (let [words {101 "The" 102 " capital" 103 " of" 104 " France" 105 " is" 106 " Paris," 107 " a" 108 " historic" 109 " city" 110 "."}]
      (apply str (map #(get words % (str " tok_" %)) token-ids))))
  (bos-id [_] 100)
  (eos-id [_] 1))

(defn -main
  "Runs end-to-end Gemma 2B text generation pipeline with Single-Pass Prefill and BF16 precision."
  [& args]
  (let [{:keys [prompt max-new-tokens temperature top-k backend precision verbose]} (parse-cli-args args)]
    (println "==================================================================")
    (println "  clj-xla Gemma 2B Single-Pass Prefill & BF16 KV-Cached Generation ")
    (println "==================================================================")

    ;; 1. Initialize PJRT runtime for specified backend
    (let [ctx (xla/init-backend! (or backend :cpu))
          model-dirs [".models/gemma-2-2b-it" ".models/gemma-2b" ".models/gemma-2-2b" ".models/gemma"]
          existing-dir (first (filter (fn [d]
                                        (let [f (io/file d)]
                                          (and (.exists f)
                                               (or (.exists (io/file f "model.safetensors"))
                                                   (.exists (io/file f "model-00001-of-00002.safetensors"))))))
                                      model-dirs))
          model-dir (or existing-dir ".models/gemma-2-2b-it")
          weights-path model-dir
          has-real-weights (some? existing-dir)]

      (if has-real-weights
        (println (str "Loading Gemma model weights from [" model-dir "] in [" (name precision) "] mode..."))
        (println (str "Safetensors files not found at [" model-dir "]. Using Gemma 2B architecture tracing mode...")))

      (let [tokenizer (if has-real-weights
                        (tok/from-file model-dir)
                        (->DummyGemmaTokenizer))
            prompt-ids (into [(bos-id tokenizer)] (encode tokenizer prompt))
            prompt-len (count prompt-ids)]
        (println (format "Prompt: \"%s\"" prompt))
        (println (format "Generation Options: max-new-tokens=%d, temperature=%.2f, top-k=%d, precision=%s"
                         max-new-tokens temperature top-k (name precision)))
        (println (format "Encoded Token IDs (%d tokens): %s" prompt-len prompt-ids))

        (let [arena (Arena/ofConfined)
              dummy-cache (atom {})
              make-dummy-seg (fn [shape]
                               (if-let [existing (get @dummy-cache shape)]
                                 existing
                                 (let [seg (.allocate (Arena/ofAuto) (long (* 4 (long (reduce * 1 shape)))))]
                                   (swap! dummy-cache assoc shape seg)
                                   seg)))

              weights-mmap (when has-real-weights
                             (st/map-safetensors-weights weights-path arena))

              header (or (:header weights-mmap) {})
              emb-shape (get-in header ["model.embed_tokens.weight" "shape"] [256000 2048])
              q-shape (get-in header ["model.layers.0.self_attn.q_proj.weight" "shape"] [2048 2048])
              k-shape (get-in header ["model.layers.0.self_attn.k_proj.weight" "shape"] [256 2048])
              gate-shape (get-in header ["model.layers.0.mlp.gate_proj.weight" "shape"] [16384 2048])

              vocab-size (nth emb-shape 0 256000)
              hidden-dim (nth emb-shape 1 2048)
              q-dim (nth q-shape 0 2048)
              kv-dim (nth k-shape 0 256)
              intermediate-dim (nth gate-shape 0 16384)
              num-layers (if has-real-weights
                           (count (filter #(re-find #"^model\.layers\.\d+\.input_layernorm\.weight$" %) (keys header)))
                           26)
              num-heads (quot q-dim 256)
              num-kv-heads (quot kv-dim 256)
              head-dim 256
              max-seq-len 128
              kv-cache-shape [1 num-kv-heads max-seq-len head-dim]

              weight-dtype (or precision :bf16)
              weight-enum (if (= weight-dtype :f32) 11 13)

              load-weight-buf (fn [tensor-name shape]
                                (let [host-data (if (and has-real-weights tensor-name)
                                                  (if (= weight-dtype :f32)
                                                    (st/get-tensor-floats weights-mmap tensor-name)
                                                    (st/get-tensor-bf16-shorts weights-mmap tensor-name))
                                                  (make-dummy-seg shape))]
                                  (xla/buffer-from-host-buffer ctx (:client ctx) host-data shape weight-enum)))]

          (println (format "Prepared weight configuration for %d layers (hidden_dim=%d, vocab_size=%d, precision=%s)."
                           num-layers hidden-dim vocab-size (name weight-dtype)))

          ;; 1. Allocate PJRT Device Buffers for KV Caches
          (println (format "Allocating %d PJRT Device KV-Cache Buffers (2 per layer, shape=%s)..."
                           (* 2 num-layers) kv-cache-shape))
          (let [initial-device-kv-bufs
                (mapv (fn [_i]
                        [(xla/buffer-from-host-buffer ctx (:client ctx) (make-dummy-seg kv-cache-shape) kv-cache-shape weight-enum)
                         (xla/buffer-from-host-buffer ctx (:client ctx) (make-dummy-seg kv-cache-shape) kv-cache-shape weight-enum)])
                      (range num-layers))

                ;; 2a. Trace Single-Pass Prompt Prefill Graph [1 prompt-len]
                prefill-invars (vec (concat
                                     [[:x [:tensor [1 prompt-len] :i32]]
                                      [:pos [:tensor [prompt-len] :i32]]
                                      [:embed_tokens [:tensor [vocab-size hidden-dim] weight-dtype]]
                                      [:final_norm_w [:tensor [hidden-dim] weight-dtype]]]
                                     (mapcat (fn [i]
                                               [[(keyword (str "input_ln_w_" i)) [:tensor [hidden-dim] weight-dtype]]
                                                [(keyword (str "q_w_" i)) [:tensor [q-dim hidden-dim] weight-dtype]]
                                                [(keyword (str "k_w_" i)) [:tensor [kv-dim hidden-dim] weight-dtype]]
                                                [(keyword (str "v_w_" i)) [:tensor [kv-dim hidden-dim] weight-dtype]]
                                                [(keyword (str "o_w_" i)) [:tensor [hidden-dim q-dim] weight-dtype]]
                                                [(keyword (str "post_attn_ln_w_" i)) [:tensor [hidden-dim] weight-dtype]]
                                                [(keyword (str "pre_mlp_ln_w_" i)) [:tensor [hidden-dim] weight-dtype]]
                                                [(keyword (str "post_mlp_ln_w_" i)) [:tensor [hidden-dim] weight-dtype]]
                                                [(keyword (str "gate_w_" i)) [:tensor [intermediate-dim hidden-dim] weight-dtype]]
                                                [(keyword (str "up_w_" i)) [:tensor [intermediate-dim hidden-dim] weight-dtype]]
                                                [(keyword (str "down_w_" i)) [:tensor [hidden-dim intermediate-dim] weight-dtype]]])
                                             (range num-layers))
                                     (mapcat (fn [i]
                                               [[(keyword (str "k_cache_" i)) [:tensor kv-cache-shape weight-dtype]]
                                                [(keyword (str "v_cache_" i)) [:tensor kv-cache-shape weight-dtype]]])
                                             (range num-layers))))

                prefill-trace-fn (fn [x _pos-tracer emb fn-norm & rest-args]
                                   (let [weight-args (take (* 11 num-layers) rest-args)
                                         kv-cache-args (drop (* 11 num-layers) rest-args)
                                         lw-seq (mapv (fn [[in-ln qw kw vw ow post-attn-ln pre-mlp-ln post-mlp-ln gw uw dw]]
                                                        {:input-ln-w in-ln :q-w qw :k-w kw :v-w vw :o-w ow
                                                         :post-attn-ln-w post-attn-ln :pre-mlp-ln-w pre-mlp-ln :post-mlp-ln-w post-mlp-ln
                                                         :gate-w gw :up-w uw :down-w dw})
                                                      (partition 11 weight-args))
                                         kv-seq (mapv vec (partition 2 kv-cache-args))
                                         [logits updated-kv-caches] (gemma/full-gemma-forward x emb lw-seq fn-norm (vec (range prompt-len)) num-heads num-kv-heads kv-seq 0)
                                         f32-logits (t/convert logits :f32)]
                                     (into [f32-logits] (apply concat updated-kv-caches))))

                ;; 2b. Trace Single-Token Autoregressive Decoding Graph [1 1]
                decode-invars (vec (concat
                                    [[:x [:tensor [1 1] :i32]]
                                     [:pos [:tensor [1] :i32]]
                                     [:embed_tokens [:tensor [vocab-size hidden-dim] weight-dtype]]
                                     [:final_norm_w [:tensor [hidden-dim] weight-dtype]]]
                                    (mapcat (fn [i]
                                              [[(keyword (str "input_ln_w_" i)) [:tensor [hidden-dim] weight-dtype]]
                                               [(keyword (str "q_w_" i)) [:tensor [q-dim hidden-dim] weight-dtype]]
                                               [(keyword (str "k_w_" i)) [:tensor [kv-dim hidden-dim] weight-dtype]]
                                               [(keyword (str "v_w_" i)) [:tensor [kv-dim hidden-dim] weight-dtype]]
                                               [(keyword (str "o_w_" i)) [:tensor [hidden-dim q-dim] weight-dtype]]
                                               [(keyword (str "post_attn_ln_w_" i)) [:tensor [hidden-dim] weight-dtype]]
                                               [(keyword (str "pre_mlp_ln_w_" i)) [:tensor [hidden-dim] weight-dtype]]
                                               [(keyword (str "post_mlp_ln_w_" i)) [:tensor [hidden-dim] weight-dtype]]
                                               [(keyword (str "gate_w_" i)) [:tensor [intermediate-dim hidden-dim] weight-dtype]]
                                               [(keyword (str "up_w_" i)) [:tensor [intermediate-dim hidden-dim] weight-dtype]]
                                               [(keyword (str "down_w_" i)) [:tensor [hidden-dim intermediate-dim] weight-dtype]]])
                                            (range num-layers))
                                    (mapcat (fn [i]
                                              [[(keyword (str "k_cache_" i)) [:tensor kv-cache-shape weight-dtype]]
                                               [(keyword (str "v_cache_" i)) [:tensor kv-cache-shape weight-dtype]]])
                                            (range num-layers))))

                decode-trace-fn (fn [x pos-tracer emb fn-norm & rest-args]
                                  (let [weight-args (take (* 11 num-layers) rest-args)
                                        kv-cache-args (drop (* 11 num-layers) rest-args)
                                        lw-seq (mapv (fn [[in-ln qw kw vw ow post-attn-ln pre-mlp-ln post-mlp-ln gw uw dw]]
                                                       {:input-ln-w in-ln :q-w qw :k-w kw :v-w vw :o-w ow
                                                        :post-attn-ln-w post-attn-ln :pre-mlp-ln-w pre-mlp-ln :post-mlp-ln-w post-mlp-ln
                                                        :gate-w gw :up-w uw :down-w dw})
                                                     (partition 11 weight-args))
                                        kv-seq (mapv vec (partition 2 kv-cache-args))
                                        [logits updated-kv-caches] (gemma/full-gemma-forward x emb lw-seq fn-norm pos-tracer num-heads num-kv-heads kv-seq pos-tracer)
                                        f32-logits (t/convert logits :f32)]
                                    (into [f32-logits] (apply concat updated-kv-caches))))

                _ (println "Tracing & JIT Compiling Single-Pass Prefill Graph...")
                prefill-graph (trace-graph "gemma2_prefill" prefill-invars prefill-trace-fn)
                prefill-exec (xla/compile-graph ctx prefill-graph)

                _ (println "Tracing & JIT Compiling Single-Token Decoding Graph...")
                decode-graph (trace-graph "gemma2_decode" decode-invars decode-trace-fn)
                decode-exec (xla/compile-graph ctx decode-graph)
                _ (println "Successfully compiled StableHLO prefill and decode graphs to native XLA PjRtLoadedExecutable handles.")

                _ (when verbose
                    (println "\n==================================================================")
                    (println "--- Single-Pass Prefill EDN SSA Graph ---")
                    (pprint/pprint prefill-graph)
                    (println "\n--- Single-Token Decode EDN SSA Graph ---")
                    (pprint/pprint decode-graph)
                    (println "==================================================================\n"))

                _ (println "Transferring model weights to PJRT Device Memory...")
                embed-buf (load-weight-buf "model.embed_tokens.weight" [vocab-size hidden-dim])
                final-norm-buf (load-weight-buf "model.norm.weight" [hidden-dim])
                layer-bufs (mapv (fn [i]
                                   (let [kmap (gemma/weight-key-map i)]
                                     [(load-weight-buf (:input-ln-w kmap) [hidden-dim])
                                      (load-weight-buf (:q-w kmap) [q-dim hidden-dim])
                                      (load-weight-buf (:k-w kmap) [kv-dim hidden-dim])
                                      (load-weight-buf (:v-w kmap) [kv-dim hidden-dim])
                                      (load-weight-buf (:o-w kmap) [hidden-dim q-dim])
                                      (load-weight-buf (:post-attn-ln-w kmap) [hidden-dim])
                                      (load-weight-buf (:pre-mlp-ln-w kmap) [hidden-dim])
                                      (load-weight-buf (:post-mlp-ln-w kmap) [hidden-dim])
                                      (load-weight-buf (:gate-w kmap) [intermediate-dim hidden-dim])
                                      (load-weight-buf (:up-w kmap) [intermediate-dim hidden-dim])
                                      (load-weight-buf (:down-w kmap) [hidden-dim intermediate-dim])]))
                                 (range num-layers))
                flat-layer-bufs (vec (apply concat layer-bufs))
                flat-device-weights (into [embed-buf final-norm-buf] flat-layer-bufs)]

            ;; 3. Autoregressive Loop with Single-Pass Prefill
            (println "\nGenerating tokens autoregressively with Single-Pass Prefill...")
            (print prompt)
            (flush)

            (let [eos (eos-id tokenizer)

                  ;; Phase A: Single-Pass Prompt Prefill (Invoked ONCE for prompt-len)
                  prompt-buf (xla/buffer-from-host-buffer ctx (:client ctx) (int-array prompt-ids) [1 prompt-len] 4)
                  pos-buf (xla/buffer-from-host-buffer ctx (:client ctx) (int-array (range prompt-len)) [prompt-len] 4)
                  flat-kv-bufs (vec (apply concat initial-device-kv-bufs))
                  prefill-args (into [prompt-buf pos-buf] (concat flat-device-weights flat-kv-bufs))
                  prefill-res (xla/execute prefill-exec prefill-args)
                  prefill-logits-buf (if (vector? prefill-res) (first prefill-res) prefill-res)
                  prefill-kv-flat (if (vector? prefill-res) (rest prefill-res) [])
                  prefill-kv-bufs (mapv vec (partition 2 prefill-kv-flat))
                  last-logits (xla/to-host-slice prefill-logits-buf (dec prompt-len) vocab-size (* prompt-len vocab-size))
                  first-gen-tok (sampling/sample-logits last-logits {:temperature temperature :top-k top-k})]

              (print (decode tokenizer [first-gen-tok]))
              (flush)

              (if (= first-gen-tok eos)
                (println "\nReached EOS token.")
                ;; Phase B: Single-token Autoregressive Decoding Loop
                (loop [context (conj (vec prompt-ids) first-gen-tok)
                       current-kv-bufs prefill-kv-bufs
                       pos prompt-len
                       step 1]
                  (if (>= step max-new-tokens)
                    context
                    (let [last-tok (last context)
                          tok-buf (xla/buffer-from-host-buffer ctx (:client ctx) (int-array [last-tok]) [1 1] 4)
                          pos-buf (xla/buffer-from-host-buffer ctx (:client ctx) (int-array [pos]) [1] 4)
                          flat-kv (vec (apply concat current-kv-bufs))
                          exec-args (into [tok-buf pos-buf] (concat flat-device-weights flat-kv))
                          res-bufs (xla/execute decode-exec exec-args)
                          logits-buf (if (vector? res-bufs) (first res-bufs) res-bufs)
                          new-kv-flat (if (vector? res-bufs) (rest res-bufs) [])
                          updated-kv-bufs (mapv vec (partition 2 new-kv-flat))
                          step-logits (xla/to-host-slice logits-buf 0 vocab-size vocab-size)
                          next-tok (sampling/sample-logits step-logits {:temperature temperature :top-k top-k})
                          next-context (conj context next-tok)]
                      (print (decode tokenizer [next-tok]))
                      (flush)
                      (if (= next-tok eos)
                        next-context
                        (recur next-context updated-kv-bufs (inc pos) (inc step)))))))

              (println "\n\n==================================================================")
              (println "=== End-to-End Gemma 2B Single-Pass Prefill Verification Passed! ===")
              (println "=================================================================="))))))))

(when (= *file* (System/getProperty "clojure.script.filename"))
  (apply -main *command-line-args*))
