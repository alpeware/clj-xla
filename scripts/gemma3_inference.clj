(ns scripts.gemma3-inference
  "Top-level runnable integration script and REPL API for end-to-end Gemma 3 text generation via pure XLA execution."
  (:require [clj-xla.core :as xla]
            [clj-xla.logic.lower :as lower]
            [clj-xla.logic.models.gemma3 :as gemma3-logic]
            [clj-xla.models.gemma :as gemma]
            [clj-xla.nn.norm :as norm]
            [clj-xla.safetensors :as st]
            [clj-xla.sampling :as sampling]
            [clj-xla.tensor :as t]
            [clj-xla.tokenizer.core :as tok]
            [clj-xla.tokenizer.protocol :refer [bos-id decode encode eos-id]]
            [clj-xla.trace :refer [trace-graph]]
            [clojure.java.io :as io])
  (:import [java.lang.foreign Arena]))

(def DEFAULT_CLI_OPTS
  {:prompt "The capital of France is"
   :max-new-tokens 10
   :temperature 0.7
   :top-k 10
   :backend :cpu
   :precision :bf16
   :method :tensor-logic
   :compare false
   :chat nil
   :verbose false})

(def DEFAULT_MODEL_DIRS
  [".models/gemma-3-270m-it" ".models/gemma-3-270m" ".models/gemma-3" ".models/gemma"])

(defn parse-cli-args
  "Parses command-line flags (--prompt, --max-new-tokens, --temperature, --top-k, --backend, --precision, --method, --compare, --chat, --model-dir, --verbose)."
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

          (and (= flag "--method") val)
          (recur (subvec remaining 2) (assoc opts :method (keyword val)))

          (and (= flag "--compare") val)
          (recur (subvec remaining 2) (assoc opts :compare (Boolean/parseBoolean val)))

          (and (= flag "--chat") val)
          (recur (subvec remaining 2) (assoc opts :chat (Boolean/parseBoolean val)))

          (and (= flag "--model-dir") val)
          (recur (subvec remaining 2) (assoc opts :model-dir val))

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
  "Initializes PJRT runtime, loads safetensors weights, and prepares model configuration for Gemma 3 REPL/CLI sessions.
   Returns an inference session map."
  ([opts]
   (let [opts (merge DEFAULT_CLI_OPTS opts)
         {:keys [backend precision model-dir]} opts
         ctx (xla/init-backend! (or backend :cpu))
         dirs (if model-dir (into [model-dir] DEFAULT_MODEL_DIRS) DEFAULT_MODEL_DIRS)
         resolved-dir (find-model-dir dirs)
         arena (Arena/ofConfined)
         weights-mmap (st/map-safetensors-weights resolved-dir arena)
         tokenizer (tok/from-file resolved-dir)
         header (or (:header weights-mmap) {})
         emb-shape (get-in header ["model.embed_tokens.weight" "shape"] [262144 640])
         q-shape (get-in header ["model.layers.0.self_attn.q_proj.weight" "shape"] [1024 640])
         k-shape (get-in header ["model.layers.0.self_attn.k_proj.weight" "shape"] [256 640])
         gate-shape (get-in header ["model.layers.0.mlp.gate_proj.weight" "shape"] [2048 640])

         vocab-size (nth emb-shape 0 262144)
         hidden-dim (nth emb-shape 1 640)
         q-dim (nth q-shape 0 1024)
         kv-dim (nth k-shape 0 256)
         intermediate-dim (nth gate-shape 0 2048)
         num-layers (count (filter #(re-find #"^model\.layers\.\d+\.input_layernorm\.weight$" %) (keys header)))
         head-dim 256
         num-heads (quot q-dim head-dim)
         num-kv-heads (quot kv-dim head-dim)
         max-seq-len 128
         weight-dtype (or precision :bf16)
         weight-enum (if (= weight-dtype :f32) 11 13)]

     (println (format "Loaded Gemma 3 model weights from [%s] in [%s] precision (%d layers)."
                      resolved-dir (name weight-dtype) num-layers))

     {:ctx ctx
      :opts opts
      :model-dir resolved-dir
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
               :weight-dtype weight-dtype
               :weight-enum weight-enum}})))

(defn allocate-device-weights
  "Transfers all model layer, embedding, and QK norm weights to PJRT device memory."
  [{:keys [ctx weights-mmap config]}]
  (let [{:keys [vocab-size hidden-dim q-dim kv-dim intermediate-dim head-dim num-layers weight-dtype weight-enum]} config
        load-fn (fn [name shape] (load-weight-buffer ctx weights-mmap name shape weight-dtype weight-enum))
        embed-buf (load-fn "model.embed_tokens.weight" [vocab-size hidden-dim])
        final-norm-buf (load-fn "model.norm.weight" [hidden-dim])
        layer-bufs (mapv (fn [i]
                           (let [kmap (gemma/gemma3-weight-key-map i)]
                             [(load-fn (:input-ln-w kmap) [hidden-dim])
                              (load-fn (:q-w kmap) [q-dim hidden-dim])
                              (load-fn (:k-w kmap) [kv-dim hidden-dim])
                              (load-fn (:v-w kmap) [kv-dim hidden-dim])
                              (load-fn (:o-w kmap) [hidden-dim q-dim])
                              (load-fn (:q-norm-w kmap) [head-dim])
                              (load-fn (:k-norm-w kmap) [head-dim])
                              (load-fn (:post-attn-ln-w kmap) [hidden-dim])
                              (load-fn (:pre-mlp-ln-w kmap) [hidden-dim])
                              (load-fn (:post-mlp-ln-w kmap) [hidden-dim])
                              (load-fn (:gate-w kmap) [intermediate-dim hidden-dim])
                              (load-fn (:up-w kmap) [intermediate-dim hidden-dim])
                              (load-fn (:down-w kmap) [hidden-dim intermediate-dim])]))
                         (range num-layers))
        flat-layer-bufs (vec (apply concat layer-bufs))]
    (into [embed-buf final-norm-buf] flat-layer-bufs)))

(defn format-prompt
  "Encodes prompt text with optional Chat ML instruction formatting for instruction-tuned (-it) models."
  [tokenizer prompt chat?]
  (if chat?
    (vec (concat [(bos-id tokenizer) 105 2364 107]
                 (encode tokenizer prompt)
                 [106 107 105 4368 107]))
    (into [(bos-id tokenizer)] (encode tokenizer prompt))))

(defn- prepare-input-tensor
  "Pads token IDs sequence to `max-len` with zero padding."
  [tokens max-len]
  (let [padded (take max-len (concat tokens (repeat 0)))]
    (int-array (vec padded))))

(defn compile-executable
  "Compiles Gemma 3 model for sequence length `max-seq-len` using either `:tensor-logic` (AST lowering) or `:trace`."
  [{:keys [ctx config]} max-seq-len comp-method]
  (let [{:keys [vocab-size hidden-dim q-dim kv-dim intermediate-dim head-dim num-layers weight-dtype]} config
        invars (vec (concat
                     [[:x [:tensor [1 max-seq-len] :i32]]
                      [:embed_tokens [:tensor [vocab-size hidden-dim] weight-dtype]]
                      [:final_norm_w [:tensor [hidden-dim] weight-dtype]]]
                     (mapcat (fn [i]
                               [[(keyword (str "input_ln_w_" i)) [:tensor [hidden-dim] weight-dtype]]
                                [(keyword (str "q_w_" i)) [:tensor [q-dim hidden-dim] weight-dtype]]
                                [(keyword (str "k_w_" i)) [:tensor [kv-dim hidden-dim] weight-dtype]]
                                [(keyword (str "v_w_" i)) [:tensor [kv-dim hidden-dim] weight-dtype]]
                                [(keyword (str "o_w_" i)) [:tensor [hidden-dim q-dim] weight-dtype]]
                                [(keyword (str "q_norm_w_" i)) [:tensor [head-dim] weight-dtype]]
                                [(keyword (str "k_norm_w_" i)) [:tensor [head-dim] weight-dtype]]
                                [(keyword (str "post_attn_ln_w_" i)) [:tensor [hidden-dim] weight-dtype]]
                                [(keyword (str "pre_mlp_ln_w_" i)) [:tensor [hidden-dim] weight-dtype]]
                                [(keyword (str "post_mlp_ln_w_" i)) [:tensor [hidden-dim] weight-dtype]]
                                [(keyword (str "gate_w_" i)) [:tensor [intermediate-dim hidden-dim] weight-dtype]]
                                [(keyword (str "up_w_" i)) [:tensor [intermediate-dim hidden-dim] weight-dtype]]
                                [(keyword (str "down_w_" i)) [:tensor [hidden-dim intermediate-dim] weight-dtype]]])
                             (range num-layers))))]
    (case comp-method
      :trace
      (do
        (println "Tracing full Gemma 3 model graph (legacy tracer)...")
        (let [trace-fn (fn [x emb fn-norm & rest-args]
                         (let [scale-factor (Math/sqrt (double hidden-dim))
                               tok-embed (t/* (t/gather emb x) scale-factor)
                               lw-seq (mapv (fn [i [in-ln qw kw vw ow qn kn post-attn pre-mlp post-mlp gw uw dw]]
                                              (let [is-global? (zero? (mod (inc i) 6))]
                                                {:input-ln-w in-ln :q-w qw :k-w kw :v-w vw :o-w ow
                                                 :q-norm-w qn :k-norm-w kn
                                                 :post-attn-ln-w post-attn :pre-mlp-ln-w pre-mlp :post-mlp-ln-w post-mlp
                                                 :gate-w gw :up-w uw :down-w dw
                                                 :norm-fn norm/gemma-rms-norm
                                                 :scale (/ 1.0 (Math/sqrt (double head-dim)))
                                                 :theta-base (if is-global? 1000000.0 10000.0)
                                                 :sliding-window (if is-global? nil 512)}))
                                            (range num-layers)
                                            (partition 13 rest-args))
                               h (reduce (fn [curr lw]
                                           (gemma/gemma-block curr lw 4 1 (vec (range max-seq-len)) nil nil))
                                         tok-embed
                                         lw-seq)
                               normed (norm/gemma-rms-norm h fn-norm 1e-6)
                               logits (t/matmul normed (t/transpose emb [1 0]))]
                           (t/convert logits weight-dtype)))
              graph (trace-graph "gemma3_trace" invars trace-fn)]
          (xla/compile-graph ctx graph)))

      :tensor-logic
      (do
        (println "Lowering declarative Tensor Logic Gemma 3 AST to StableHLO graph...")
        (let [cfg (assoc config :max-seq-len max-seq-len :head-dim head-dim)
              ast (gemma3-logic/gemma3-model-ast cfg)
              graph (lower/ast->graph "gemma3_logic" invars ast #{:logits})]
          (xla/compile-graph ctx graph))))))

(defn generate-text
  "Top-level REPL/programmatic helper: runs full end-to-end text generation on an initialized session."
  ([session] (generate-text session (or (:prompt (:opts session)) "The capital of France is")))
  ([session prompt-text]
   (let [{:keys [ctx tokenizer config opts]} session
         {:keys [max-new-tokens temperature top-k method compare chat]} opts
         {:keys [vocab-size weight-dtype]} config
         chat? (if (some? chat) (boolean chat) (boolean (re-find #"-it" (or (:model-dir session) ""))))
         prompt-ids (format-prompt tokenizer prompt-text chat?)
         prompt-len (count prompt-ids)
         max-seq-len (max 32 (+ prompt-len max-new-tokens 4))
         device-weights (allocate-device-weights session)]

     (println (format "Prompt: \"%s\" (chat=%s)" prompt-text chat?))
     (println (format "Generation Options: max-new-tokens=%d, temperature=%.2f, top-k=%d, method=%s, compare=%s"
                      max-new-tokens temperature top-k method compare))
     (println (format "Encoded Token IDs (%d tokens): %s" prompt-len prompt-ids))

     (if compare
       (do
         (println "\n==================================================================")
         (println "               STARTING E2E PARITY COMPARISON MODE                ")
         (println "==================================================================")
         (let [exec-trace (compile-executable session max-seq-len :trace)
               exec-logic (compile-executable session max-seq-len :tensor-logic)
               input-arr (prepare-input-tensor prompt-ids max-seq-len)
               input-buf (xla/buffer-from-host-buffer ctx (:client ctx) input-arr [1 max-seq-len] 4)
               args (into [input-buf] device-weights)

               _ (println "\nExecuting prompt forward pass on both engines...")
               out-trace (xla/execute exec-trace args)
               out-logic (xla/execute exec-logic args)
               buf-t (if (vector? out-trace) (first out-trace) out-trace)
               buf-l (if (vector? out-logic) (first out-logic) out-logic)

               slice-t (xla/to-host-slice buf-t (dec prompt-len) vocab-size (* max-seq-len vocab-size) weight-dtype)
               slice-l (xla/to-host-slice buf-l (dec prompt-len) vocab-size (* max-seq-len vocab-size) weight-dtype)

               diffs (mapv (fn [a b] (Math/abs (double (- a b)))) (vec slice-t) (vec slice-l))
               max-diff (reduce max 0.0 diffs)
               mean-diff (/ (reduce + 0.0 diffs) (count diffs))
               top-k-trace (take 5 (sort-by second > (map-indexed vector slice-t)))
               top-k-logic (take 5 (sort-by second > (map-indexed vector slice-l)))]

           (println "\n--- Logit Parity Metrics ---")
           (println (format "Max Absolute Error  : %.6e" max-diff))
           (println (format "Mean Absolute Error : %.6e" mean-diff))
           (println "Top 5 Logits (Tracer)      :" top-k-trace)
           (println "Top 5 Logits (Tensor Logic):" top-k-logic)

           (println "\n--- Running Autoregressive Generation Comparison ---")
           (let [generate-tokens-fn (fn [engine-name exec]
                                      (println (str "\nGenerating with [" engine-name "]..."))
                                      (print prompt-text)
                                      (flush)
                                      (let [cur-tokens (atom (vec prompt-ids))
                                            eos (eos-id tokenizer)]
                                        (dotimes [_ max-new-tokens]
                                          (let [s-len (count @cur-tokens)
                                                in-arr (prepare-input-tensor @cur-tokens max-seq-len)
                                                in-b (xla/buffer-from-host-buffer ctx (:client ctx) in-arr [1 max-seq-len] 4)
                                                in-args (into [in-b] device-weights)
                                                out (xla/execute exec in-args)
                                                out-b (if (vector? out) (first out) out)
                                                l (xla/to-host-slice out-b (dec s-len) vocab-size (* max-seq-len vocab-size) weight-dtype)
                                                next-id (sampling/sample-logits l {:temperature temperature :top-k top-k})]
                                            (swap! cur-tokens conj next-id)
                                            (print (decode tokenizer [next-id]))
                                            (flush)
                                            (when (= next-id eos)
                                              (reduced nil))))
                                        (println)
                                        @cur-tokens))
                 tokens-trace (generate-tokens-fn "Legacy Tracer" exec-trace)
                 tokens-logic (generate-tokens-fn "Tensor Logic" exec-logic)
                 str-trace (decode tokenizer tokens-trace)
                 str-logic (decode tokenizer tokens-logic)
                 match? (= tokens-trace tokens-logic)]

             (println "\n==================================================================")
             (println "                   FINAL VERIFICATION SUMMARY                     ")
             (println "==================================================================")
             (println "Legacy Tracer Output:\n" str-trace)
             (println "\nTensor Logic Output :\n" str-logic)
             (println "Token Matching:" (if match? "EXACT MATCH (100% PARITY)" "CLOSE PARITY"))
             tokens-logic)))

       ;; Standard generation with selected method
       (let [exec (compile-executable session max-seq-len (or method :tensor-logic))
             eos (eos-id tokenizer)
             cur-tokens (atom (vec prompt-ids))]
         (println "Successfully compiled model to native XLA PjRtLoadedExecutable handle.")
         (println "\nGenerating tokens autoregressively...")
         (print prompt-text)
         (flush)
         (dotimes [_ max-new-tokens]
           (let [s-len (count @cur-tokens)
                 in-arr (prepare-input-tensor @cur-tokens max-seq-len)
                 in-b (xla/buffer-from-host-buffer ctx (:client ctx) in-arr [1 max-seq-len] 4)
                 in-args (into [in-b] device-weights)
                 out (xla/execute exec in-args)
                 out-b (if (vector? out) (first out) out)
                 l (xla/to-host-slice out-b (dec s-len) vocab-size (* max-seq-len vocab-size) weight-dtype)
                 next-id (sampling/sample-logits l {:temperature temperature :top-k top-k})]
             (swap! cur-tokens conj next-id)
             (print (decode tokenizer [next-id]))
             (flush)
             (when (= next-id eos)
               (println "\nReached EOS token.")
               (reduced nil))))
         (println "\n\n==================================================================")
         (println "Final Generated Sequence:")
         (println (decode tokenizer @cur-tokens))
         (println "==================================================================")
         (println "=== End-to-End Gemma 3 Generation Completed! ===")
         @cur-tokens)))))

(defn -main
  "CLI entrypoint for Gemma 3 text generation."
  [& args]
  (let [opts (parse-cli-args args)]
    (println "==================================================================")
    (println "      clj-xla Gemma 3 270M End-to-End Text Generation Loop        ")
    (println "==================================================================")
    (let [session (init-inference-session opts)]
      (generate-text session (:prompt opts)))))

(when (= *file* (System/getProperty "clojure.script.filename"))
  (apply -main *command-line-args*))
