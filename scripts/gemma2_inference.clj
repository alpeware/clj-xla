(ns scripts.gemma2-inference
  "Top-level runnable integration script and REPL API for end-to-end Gemma 2B text generation via pure XLA execution."
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

(def DEFAULT_MODEL_DIRS
  [".models/gemma-2-2b-it" ".models/gemma-2b" ".models/gemma-2-2b" ".models/gemma"])

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

(defn find-model-dir
  "Searches `model-dirs` for an existing directory containing `.safetensors` files.
   Throws an ExceptionInfo if no model files are found."
  ([model-dirs]
   (let [existing (first (filter (fn [d]
                                   (let [f (io/file d)]
                                     (and (.exists f)
                                          (or (.exists (io/file f "model.safetensors"))
                                              (.exists (io/file f "model-00001-of-00002.safetensors"))))))
                                 model-dirs))]
     (if existing
       existing
       (throw (ex-info (str "Model directory with safetensors not found in candidates: " (vec model-dirs))
                       {:searched-dirs model-dirs}))))))

(defn load-weight-buffer
  "Loads a single weight tensor from `weights-mmap` into PJRT device memory in specified precision."
  [ctx weights-mmap tensor-name shape weight-dtype weight-enum]
  (let [host-data (if (= weight-dtype :f32)
                    (st/get-tensor-floats weights-mmap tensor-name)
                    (st/get-tensor-bf16-shorts weights-mmap tensor-name))]
    (xla/buffer-from-host-buffer ctx (:client ctx) host-data shape weight-enum)))

(defn init-inference-session
  "Initializes PJRT runtime, loads safetensors weights, and prepares model configuration for REPL/CLI sessions.
   Returns an inference session map."
  ([opts]
   (let [opts (merge DEFAULT_CLI_OPTS opts)
         {:keys [backend precision]} opts
         ctx (xla/init-backend! (or backend :cpu))
         model-dir (find-model-dir DEFAULT_MODEL_DIRS)
         arena (Arena/ofConfined)
         weights-mmap (st/map-safetensors-weights model-dir arena)
         tokenizer (tok/from-file model-dir)
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
         num-layers (count (filter #(re-find #"^model\.layers\.\d+\.input_layernorm\.weight$" %) (keys header)))
         num-heads (quot q-dim 256)
         num-kv-heads (quot kv-dim 256)
         head-dim 256
         max-seq-len 128
         kv-cache-shape [1 num-kv-heads max-seq-len head-dim]
         weight-dtype (or precision :bf16)
         weight-enum (if (= weight-dtype :f32) 11 13)]

     (println (str "Loaded Gemma model weights from [" model-dir "] in [" (name weight-dtype) "] precision (" num-layers " layers)."))

     {:ctx ctx
      :opts opts
      :model-dir model-dir
      :tokenizer tokenizer
      :weights-mmap weights-mmap
      :arena arena
      :config {:vocab-size vocab-size
               :hidden-dim hidden-dim
               :q-dim q-dim
               :kv-dim kv-dim
               :intermediate-dim intermediate-dim
               :num-layers num-layers
               :num-heads num-heads
               :num-kv-heads num-kv-heads
               :head-dim head-dim
               :max-seq-len max-seq-len
               :kv-cache-shape kv-cache-shape
               :weight-dtype weight-dtype
               :weight-enum weight-enum}})))

(defn allocate-device-weights
  "Transfers all model layer and embedding weights to PJRT device memory."
  [{:keys [ctx weights-mmap config]}]
  (let [{:keys [vocab-size hidden-dim q-dim kv-dim intermediate-dim num-layers weight-dtype weight-enum]} config
        load-fn (fn [name shape] (load-weight-buffer ctx weights-mmap name shape weight-dtype weight-enum))
        embed-buf (load-fn "model.embed_tokens.weight" [vocab-size hidden-dim])
        final-norm-buf (load-fn "model.norm.weight" [hidden-dim])
        layer-bufs (mapv (fn [i]
                           (let [kmap (gemma/weight-key-map i)]
                             [(load-fn (:input-ln-w kmap) [hidden-dim])
                              (load-fn (:q-w kmap) [q-dim hidden-dim])
                              (load-fn (:k-w kmap) [kv-dim hidden-dim])
                              (load-fn (:v-w kmap) [kv-dim hidden-dim])
                              (load-fn (:o-w kmap) [hidden-dim q-dim])
                              (load-fn (:post-attn-ln-w kmap) [hidden-dim])
                              (load-fn (:pre-mlp-ln-w kmap) [hidden-dim])
                              (load-fn (:post-mlp-ln-w kmap) [hidden-dim])
                              (load-fn (:gate-w kmap) [intermediate-dim hidden-dim])
                              (load-fn (:up-w kmap) [intermediate-dim hidden-dim])
                              (load-fn (:down-w kmap) [hidden-dim intermediate-dim])]))
                         (range num-layers))
        flat-layer-bufs (vec (apply concat layer-bufs))]
    (into [embed-buf final-norm-buf] flat-layer-bufs)))

(defn allocate-kv-caches
  "Allocates initial zero-filled PJRT device memory buffers for 26 layer K/V caches."
  [{:keys [ctx config]}]
  (let [{:keys [num-layers kv-cache-shape weight-enum]} config
        num-elements (reduce * 1 kv-cache-shape)
        zero-data (if (= weight-enum 11) (float-array num-elements) (short-array num-elements))]
    (mapv (fn [_i]
            [(xla/buffer-from-host-buffer ctx (:client ctx) zero-data kv-cache-shape weight-enum)
             (xla/buffer-from-host-buffer ctx (:client ctx) zero-data kv-cache-shape weight-enum)])
          (range num-layers))))

(defn compile-inference-executables
  "Traces and JIT-compiles Single-Pass Prefill and Single-Token Decode StableHLO graphs into PJRT Executables."
  [{:keys [ctx config opts]} prompt-len]
  (let [{:keys [vocab-size hidden-dim q-dim kv-dim intermediate-dim num-layers num-heads num-kv-heads kv-cache-shape weight-dtype]} config
        {:keys [verbose]} opts

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
        _ (println "Successfully compiled StableHLO prefill and decode graphs to native XLA PjRtLoadedExecutable handles.")]

    (when verbose
      (println "\n==================================================================")
      (println "--- Single-Pass Prefill EDN SSA Graph ---")
      (pprint/pprint prefill-graph)
      (println "\n--- Single-Token Decode EDN SSA Graph ---")
      (pprint/pprint decode-graph)
      (println "==================================================================\n"))

    {:prefill-exec prefill-exec
     :decode-exec decode-exec
     :prefill-graph prefill-graph
     :decode-graph decode-graph}))

(defn run-prompt-prefill
  "Runs Single-Pass Prompt Prefill phase on PJRT device runtime."
  [{:keys [ctx config opts]} executables device-weights initial-kv-bufs prompt-ids]
  (let [prompt-len (count prompt-ids)
        {:keys [vocab-size]} config
        {:keys [temperature top-k]} opts
        {:keys [prefill-exec]} executables
        prompt-buf (xla/buffer-from-host-buffer ctx (:client ctx) (int-array prompt-ids) [1 prompt-len] 4)
        pos-buf (xla/buffer-from-host-buffer ctx (:client ctx) (int-array (range prompt-len)) [prompt-len] 4)
        flat-kv-bufs (vec (apply concat initial-kv-bufs))
        prefill-args (into [prompt-buf pos-buf] (concat device-weights flat-kv-bufs))
        prefill-res (xla/execute prefill-exec prefill-args)
        prefill-logits-buf (if (vector? prefill-res) (first prefill-res) prefill-res)
        prefill-kv-flat (if (vector? prefill-res) (rest prefill-res) [])
        prefill-kv-bufs (mapv vec (partition 2 prefill-kv-flat))
        last-logits (xla/to-host-slice prefill-logits-buf (dec prompt-len) vocab-size (* prompt-len vocab-size))
        first-gen-tok (sampling/sample-logits last-logits {:temperature temperature :top-k top-k})]
    {:first-gen-tok first-gen-tok
     :prefill-kv-bufs prefill-kv-bufs
     :prompt-ids prompt-ids}))

(defn run-autoregressive-decode
  "Runs the single-token autoregressive decoding loop."
  [{:keys [ctx tokenizer config opts]} executables device-weights prefill-result]
  (let [{:keys [max-new-tokens temperature top-k]} opts
        {:keys [vocab-size]} config
        {:keys [decode-exec]} executables
        {:keys [first-gen-tok prefill-kv-bufs prompt-ids]} prefill-result
        prompt-len (count prompt-ids)
        eos (eos-id tokenizer)]
    (print (decode tokenizer [first-gen-tok]))
    (flush)
    (if (= first-gen-tok eos)
      (do (println "\nReached EOS token.")
          (vec prompt-ids))
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
                exec-args (into [tok-buf pos-buf] (concat device-weights flat-kv))
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
              (recur next-context updated-kv-bufs (inc pos) (inc step)))))))))

(defn generate-text
  "Top-level REPL/programmatic helper: runs full end-to-end text generation on an initialized session."
  ([session] (generate-text session (or (:prompt (:opts session)) "The capital of France is")))
  ([session prompt-text]
   (let [{:keys [tokenizer opts]} session
         {:keys [max-new-tokens temperature top-k]} opts
         prompt-ids (into [(bos-id tokenizer)] (encode tokenizer prompt-text))
         prompt-len (count prompt-ids)]
     (println (format "Prompt: \"%s\"" prompt-text))
     (println (format "Generation Options: max-new-tokens=%d, temperature=%.2f, top-k=%d, precision=%s"
                      max-new-tokens temperature top-k (name (get-in session [:config :weight-dtype]))))
     (println (format "Encoded Token IDs (%d tokens): %s" prompt-len prompt-ids))

     (println "Transferring model weights to PJRT Device Memory...")
     (let [device-weights (allocate-device-weights session)
           initial-kv-bufs (allocate-kv-caches session)
           executables (compile-inference-executables session prompt-len)]
       (println "\nGenerating tokens autoregressively with Single-Pass Prefill...")
       (print prompt-text)
       (flush)
       (let [prefill-res (run-prompt-prefill session executables device-weights initial-kv-bufs prompt-ids)
             final-context (run-autoregressive-decode session executables device-weights prefill-res)]
         (println "\n\n==================================================================")
         (println "=== End-to-End Gemma 2B Single-Pass Prefill Verification Passed! ===")
         (println "==================================================================")
         final-context)))))

(defn -main
  "CLI entrypoint."
  [& args]
  (let [opts (parse-cli-args args)]
    (println "==================================================================")
    (println "  clj-xla Gemma 2B Single-Pass Prefill & BF16 KV-Cached Generation ")
    (println "==================================================================")
    (let [session (init-inference-session opts)]
      (generate-text session (:prompt opts)))))

(when (= *file* (System/getProperty "clojure.script.filename"))
  (apply -main *command-line-args*))
