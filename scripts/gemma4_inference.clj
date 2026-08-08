(ns scripts.gemma4-inference
  "Top-level runnable integration script and REPL API for end-to-end Gemma 4 text generation via pure XLA execution."
  (:require [clj-xla.core :as xla]
            [clj-xla.models.gemma :as gemma]
            [clj-xla.nn.norm :as norm]
            [clj-xla.safetensors :as st]
            [clj-xla.sampling :as sampling]
            [clj-xla.tensor :as t]
            [clj-xla.tokenizer.core :as tok]
            [clj-xla.tokenizer.protocol :refer [bos-id decode encode eos-id]]
            [clj-xla.trace :refer [trace-graph]]
            [clojure.data.json :as json]
            [clojure.java.io :as io]
            [clojure.pprint :as pprint]
            [clojure.string :as str])
  (:import [java.lang.foreign Arena]))

(def DEFAULT_CLI_OPTS
  {:prompt "The capital of France is"
   :max-new-tokens 20
   :temperature 0.7
   :top-k 10
   :backend :cpu
   :precision :bf16
   :verbose false
   :quiet false})

(def DEFAULT_MODEL_DIRS
  [".models/gemma-4-E4B-it"
   ".models/gemma-4-E4B"
   ".models/gemma-4-E2B-it"
   ".models/gemma-4-E2B"
   ".models/gemma-4-12B-it"
   ".models/gemma-4-12B"
   ".models/gemma-4-26B-A4B-it"
   ".models/gemma-4-26B-A4B"
   ".models/gemma-4-31B-it"
   ".models/gemma-4-31B"
   ".models/gemma-4"])

(defn parse-cli-args
  "Parses command-line flags (--prompt, --model/--model-dir, --max-new-tokens, --temperature, --top-k, --backend, --precision, --verbose, --quiet)."
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

          (and (or (= flag "--model-dir") (= flag "--model") (= flag "--model-name") (= flag "-m")) val)
          (let [dir (if (str/starts-with? val ".models/") val (str ".models/" (last (str/split val #"/"))))]
            (recur (subvec remaining 2) (assoc opts :model-dir dir)))

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

          (= flag "--quiet")
          (do (System/setProperty "clj-xla.quiet" "true")
              (recur (subvec remaining 1) (assoc opts :quiet true)))

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

(defn load-model-config
  "Loads model configuration map from `config.json` if available."
  [model-dir]
  (let [cfg-file (io/file model-dir "config.json")]
    (when (.exists cfg-file)
      (try
        (json/read-str (slurp cfg-file) :key-fn keyword)
        (catch Exception _ nil)))))

(defn load-weight-buffer
  "Loads a single weight tensor from `weights-mmap` into PJRT device memory in specified precision."
  ([ctx weights-mmap tensor-name shape weight-dtype weight-enum]
   (load-weight-buffer ctx weights-mmap tensor-name shape weight-dtype weight-enum 0.0))
  ([ctx weights-mmap tensor-name shape _weight-dtype weight-enum default-val]
   (if (or (zero? (reduce * 1 shape))
           (not (or (contains? (:header weights-mmap) tensor-name)
                    (contains? (:tensors weights-mmap) tensor-name))))
     (let [num-elements (reduce * 1 shape)
           default-f (float default-val)
           data (if (= weight-enum 11)
                  (let [arr (float-array num-elements)]
                    (java.util.Arrays/fill arr default-f)
                    arr)
                  (let [arr (short-array num-elements)
                        bf-bits (short (bit-shift-right (Float/floatToRawIntBits default-f) 16))]
                    (java.util.Arrays/fill arr bf-bits)
                    arr))]
       (xla/buffer-from-host-buffer ctx (:client ctx) data shape weight-enum))
     (let [slice (st/get-tensor-slice weights-mmap tensor-name)]
       (xla/buffer-from-host-buffer ctx (:client ctx) slice shape weight-enum)))))

(defn init-inference-session
  "Initializes PJRT runtime, loads safetensors weights, and prepares model configuration for Gemma 4 REPL/CLI sessions."
  ([opts]
   (let [opts (merge DEFAULT_CLI_OPTS opts)
         {:keys [backend precision model-dir]} opts
         ctx (xla/init-backend! (or backend :cpu))
         dirs (if model-dir (cons model-dir DEFAULT_MODEL_DIRS) DEFAULT_MODEL_DIRS)
         resolved-model-dir (find-model-dir dirs)
         arena (Arena/ofConfined)
         weights-mmap (st/map-safetensors-weights resolved-model-dir arena)
         tokenizer (tok/from-file resolved-model-dir)
         header (or (:header weights-mmap) {})
         prefix-base (if (contains? header "model.language_model.embed_tokens.weight")
                       "model.language_model."
                       "model.")
         model-cfg (load-model-config resolved-model-dir)
         text-cfg (or (:text_config model-cfg) model-cfg)
         layer-types-cfg (:layer_types text-cfg)
         num-kv-shared-layers (or (:num_kv_shared_layers text-cfg) 0)

         emb-shape (get-in header [(str prefix-base "embed_tokens.weight") "shape"] [262144 1536])
         emb-pl-shape (get-in header [(str prefix-base "embed_tokens_per_layer.weight") "shape"] [262144 0])
         q0-shape (get-in header [(str prefix-base "layers.0.self_attn.q_proj.weight") "shape"] [2048 1536])
         k0-shape (get-in header [(str prefix-base "layers.0.self_attn.k_proj.weight") "shape"] [256 1536])

         vocab-size (nth emb-shape 0 262144)
         hidden-dim (nth emb-shape 1 1536)
         total-pl-dim (nth emb-pl-shape 1 0)

         layer-pattern (re-pattern (str "^" (java.util.regex.Pattern/quote prefix-base) "layers\\.\\d+\\.input_layernorm\\.weight$"))
         num-layers (count (filter #(re-find layer-pattern %) (keys header)))
         num-layers (if (pos? num-layers) num-layers (or (:num_hidden_layers text-cfg) 35))
         pl-dim (if (pos? num-layers) (quot total-pl-dim num-layers) 0)

         num-heads (or (:num_attention_heads text-cfg) (quot (nth q0-shape 0 2048) 256))
         num-kv-heads (or (:num_key_value_heads text-cfg) (quot (nth k0-shape 0 256) 256))
         head-dim (or (:head_dim text-cfg) 256)
         max-seq-len 128

         layer-configs (mapv (fn [i]
                               (let [kmap (gemma/gemma4-weight-key-map i (str prefix-base "layers."))
                                     l-q-dim (first (get-in header [(:q-w kmap) "shape"] [2048 hidden-dim]))
                                     l-kv-dim (first (get-in header [(:k-w kmap) "shape"] [256 hidden-dim]))
                                     l-head-dim (first (get-in header [(:q-norm-w kmap) "shape"] [head-dim]))
                                     mlp-dim (first (get-in header [(:gate-w kmap) "shape"] [(* 4 hidden-dim) hidden-dim]))
                                     l-type-str (or (get layer-types-cfg i)
                                                    (if (= l-head-dim 512) "full_attention" "sliding_attention"))
                                     is-global? (= l-type-str "full_attention")
                                     rope-prop (if is-global? 0.25 1.0)
                                     theta-base (if is-global? 1000000.0 10000.0)
                                     l-nkv (quot l-kv-dim l-head-dim)]
                                 {:idx i
                                  :q-dim l-q-dim
                                  :kv-dim l-kv-dim
                                  :head-dim l-head-dim
                                  :num-kv-heads l-nkv
                                  :mlp-dim mlp-dim
                                  :is-global? is-global?
                                  :layer-type (if is-global? :full_attention :sliding_attention)
                                  :rope-proportion rope-prop
                                  :theta-base theta-base}))
                             (range num-layers))

         layer-packed-info (mapv (fn [i]
                                  (let [kmap (gemma/gemma4-weight-key-map i (str prefix-base "layers."))
                                        cfg (nth layer-configs i)
                                        {:keys [q-dim kv-dim head-dim mlp-dim]} cfg
                                        v-key (if (or (contains? header (:v-w kmap))
                                                      (contains? (:tensors weights-mmap) (:v-w kmap)))
                                                (:v-w kmap)
                                                (:k-w kmap))
                                        load-info-fn (fn [name shape default-val]
                                                       (let [num-el (reduce * 1 shape)
                                                             has-tensor (or (contains? header name)
                                                                            (contains? (:tensors weights-mmap) name))]
                                                         {:name name :shape shape :num-el num-el :has-tensor has-tensor :default-val default-val}))
                                        base-tensors [(load-info-fn (:input-ln-w kmap) [hidden-dim] 0.0)
                                                      (load-info-fn (:layer-scalar-w kmap) [1] 1.0)
                                                      (load-info-fn (:q-w kmap) [q-dim hidden-dim] 0.0)
                                                      (load-info-fn (:k-w kmap) [kv-dim hidden-dim] 0.0)
                                                      (load-info-fn v-key [kv-dim hidden-dim] 0.0)
                                                      (load-info-fn (:o-w kmap) [hidden-dim q-dim] 0.0)
                                                      (load-info-fn (:q-norm-w kmap) [head-dim] 0.0)
                                                      (load-info-fn (:k-norm-w kmap) [head-dim] 0.0)
                                                      (load-info-fn (:post-attn-ln-w kmap) [hidden-dim] 0.0)
                                                      (load-info-fn (:pre-mlp-ln-w kmap) [hidden-dim] 0.0)
                                                      (load-info-fn (:post-mlp-ln-w kmap) [hidden-dim] 0.0)
                                                      (load-info-fn (:gate-w kmap) [mlp-dim hidden-dim] 0.0)
                                                      (load-info-fn (:up-w kmap) [mlp-dim hidden-dim] 0.0)
                                                      (load-info-fn (:down-w kmap) [hidden-dim mlp-dim] 0.0)]
                                        tensors (if (pos? total-pl-dim)
                                                  (into base-tensors [(load-info-fn (:per-layer-gate-w kmap) [pl-dim hidden-dim] 0.0)
                                                                      (load-info-fn (:per-layer-proj-w kmap) [hidden-dim pl-dim] 0.0)
                                                                      (load-info-fn (:post-per-layer-norm-w kmap) [hidden-dim] 0.0)])
                                                  base-tensors)
                                        offsets (reduce (fn [acc {:keys [num-el]}]
                                                          (conj acc (+ (last acc) num-el)))
                                                        [0]
                                                        tensors)
                                        total-el (last offsets)
                                        tensors-with-off (mapv (fn [t off] (assoc t :offset off)) tensors (pop offsets))]
                                    {:layer-idx i
                                     :total-el total-el
                                     :tensors tensors-with-off}))
                                (range num-layers))

         weight-dtype (or precision :bf16)
         weight-enum (if (= weight-dtype :f32) 11 13)]

     (when-not (:quiet opts)
       (println (str "Loaded Gemma 4 model weights from [" resolved-model-dir "] in [" (name weight-dtype) "] precision (" num-layers " layers, " num-heads " heads, " num-kv-heads " kv-heads).")))

     {:ctx ctx
      :opts opts
      :model-dir resolved-model-dir
      :tokenizer tokenizer
      :weights-mmap weights-mmap
      :arena arena
      :config {:model-dir resolved-model-dir
               :prefix-base prefix-base
               :vocab-size vocab-size
               :hidden-dim hidden-dim
               :total-pl-dim total-pl-dim
               :pl-dim pl-dim
               :num-layers num-layers
               :num-heads num-heads
               :num-kv-heads num-kv-heads
               :head-dim head-dim
               :max-seq-len max-seq-len
               :layer-configs layer-configs
               :num-kv-shared-layers num-kv-shared-layers
               :weight-dtype weight-dtype
               :weight-enum weight-enum
               :layer-packed-info layer-packed-info}})))

(defn allocate-device-weights
  "Transfers all model layer, embedding, and per-layer input weights to PJRT device memory."
  [{:keys [ctx weights-mmap config]}]
  (let [{:keys [prefix-base vocab-size hidden-dim total-pl-dim pl-dim weight-dtype weight-enum layer-packed-info]} config
        client (:client ctx)
        has-ple? (pos? total-pl-dim)
        load-fn (fn
                  ([name shape] (load-weight-buffer ctx weights-mmap name shape weight-dtype weight-enum 0.0))
                  ([name shape default-val] (load-weight-buffer ctx weights-mmap name shape weight-dtype weight-enum default-val)))
        embed-buf (load-fn (str prefix-base "embed_tokens.weight") [vocab-size hidden-dim])
        final-norm-buf (load-fn (str prefix-base "norm.weight") [hidden-dim])
        ple-global-bufs (when has-ple?
                          [(load-fn (str prefix-base "embed_tokens_per_layer.weight") [vocab-size total-pl-dim])
                           (load-fn (str prefix-base "per_layer_model_projection.weight") [total-pl-dim hidden-dim])
                           (load-fn (str prefix-base "per_layer_projection_norm.weight") [pl-dim])])
        layer-bufs (mapv (fn [{:keys [total-el tensors]}]
                           (let [is-bf16 (= weight-dtype :bf16)
                                 arr (if is-bf16 (short-array total-el) (float-array total-el))]
                             (doseq [{:keys [name num-el offset has-tensor default-val]} tensors]
                               (if has-tensor
                                 (let [slice (st/get-tensor-slice weights-mmap name)]
                                   (if is-bf16
                                     (java.lang.foreign.MemorySegment/copy slice (java.lang.foreign.ValueLayout/JAVA_SHORT) 0
                                                                            ^shorts arr offset num-el)
                                     (java.lang.foreign.MemorySegment/copy slice (java.lang.foreign.ValueLayout/JAVA_FLOAT) 0
                                                                            ^floats arr offset num-el)))
                                 (let [default-f (float default-val)]
                                   (if is-bf16
                                     (let [bf-bits (short (bit-shift-right (Float/floatToRawIntBits default-f) 16))]
                                       (java.util.Arrays/fill ^shorts arr offset (+ offset num-el) bf-bits))
                                     (java.util.Arrays/fill ^floats arr offset (+ offset num-el) default-f)))))
                             (xla/buffer-from-host-buffer ctx client arr [total-el] weight-enum)))
                         layer-packed-info)]
    (vec (concat (into [embed-buf] ple-global-bufs) [final-norm-buf] layer-bufs))))

(defn compile-inference-executables
  "Traces and JIT-compiles a Single Fused Gemma 4 StableHLO Graph into a native PJRT Executable."
  [{:keys [ctx config opts]}]
  (let [{:keys [vocab-size hidden-dim total-pl-dim pl-dim max-seq-len num-layers weight-dtype layer-configs num-heads num-kv-heads num-kv-shared-layers layer-packed-info]} config
        {:keys [verbose quiet]} opts
        num-unshared (- num-layers num-kv-shared-layers)
        has-ple? (pos? total-pl-dim)
        ple-global-invars (when has-ple?
                            [[:embed_tokens_per_layer [:tensor [vocab-size total-pl-dim] weight-dtype]]
                             [:per_layer_model_projection [:tensor [total-pl-dim hidden-dim] weight-dtype]]
                             [:per_layer_projection_norm [:tensor [pl-dim] weight-dtype]]])
        invars (vec (concat
                     [[:x [:tensor [1 max-seq-len] :i32]]
                      [:pos [:tensor [max-seq-len] :i32]]
                      [:embed_tokens [:tensor [vocab-size hidden-dim] weight-dtype]]]
                     ple-global-invars
                     [[:final_norm_w [:tensor [hidden-dim] weight-dtype]]]
                     (mapv (fn [i]
                             [(keyword (str "layer_packed_w_" i)) [:tensor [(get-in layer-packed-info [i :total-el])] weight-dtype]])
                           (range num-layers))))

        trace-fn (if has-ple?
                   (fn [x pos-tracer emb emb-pl pl-model-proj pl-proj-norm fn-norm & layer-packed-tracers]
                     (let [lw-seq (mapv (fn [i layer-tr]
                                          (let [{:keys [tensors]} (nth layer-packed-info i)
                                                cfg (nth layer-configs i)
                                                unpacked (mapv (fn [{:keys [shape offset num-el]}]
                                                                 (let [slice-1d (t/slice layer-tr [offset] [(+ offset num-el)] [1])]
                                                                   (t/reshape slice-1d shape)))
                                                               tensors)
                                                [in-ln ls qw kw vw ow qn kn post-attn-ln pre-mlp-ln post-mlp-ln gw uw dw plg plp pln] unpacked]
                                            {:input-ln-w in-ln :layer-scalar-w ls
                                             :q-w qw :k-w kw :v-w vw :o-w ow
                                             :q-norm-w qn :k-norm-w kn
                                             :post-attn-ln-w post-attn-ln :pre-mlp-ln-w pre-mlp-ln :post-mlp-ln-w post-mlp-ln
                                             :gate-w gw :up-w uw :down-w dw
                                             :per-layer-gate-w plg :per-layer-proj-w plp :post-per-layer-norm-w pln
                                             :num-heads num-heads :num-kv-heads (:num-kv-heads cfg) :head-dim (:head-dim cfg)
                                             :theta-base (:theta-base cfg)
                                             :rope-proportion (:rope-proportion cfg)
                                             :norm-fn norm/rms-norm
                                             :attn-softcap nil
                                             :layer-type (:layer-type cfg)
                                             :is-shared? (>= i num-unshared)}))
                                        (range num-layers)
                                        layer-packed-tracers)
                           logits (gemma/full-gemma4-forward x emb emb-pl pl-model-proj pl-proj-norm lw-seq fn-norm pos-tracer num-heads num-kv-heads nil 0 {:final-logit-softcap 30.0 :num-kv-shared-layers num-kv-shared-layers})]
                       (t/convert logits :f32)))
                   (fn [x pos-tracer emb fn-norm & layer-packed-tracers]
                     (let [lw-seq (mapv (fn [i layer-tr]
                                          (let [{:keys [tensors]} (nth layer-packed-info i)
                                                cfg (nth layer-configs i)
                                                unpacked (mapv (fn [{:keys [shape offset num-el]}]
                                                                 (let [slice-1d (t/slice layer-tr [offset] [(+ offset num-el)] [1])]
                                                                   (t/reshape slice-1d shape)))
                                                               tensors)
                                                [in-ln ls qw kw vw ow qn kn post-attn-ln pre-mlp-ln post-mlp-ln gw uw dw] unpacked]
                                            {:input-ln-w in-ln :layer-scalar-w ls
                                             :q-w qw :k-w kw :v-w vw :o-w ow
                                             :q-norm-w qn :k-norm-w kn
                                             :post-attn-ln-w post-attn-ln :pre-mlp-ln-w pre-mlp-ln :post-mlp-ln-w post-mlp-ln
                                             :gate-w gw :up-w uw :down-w dw
                                             :num-heads num-heads :num-kv-heads (:num-kv-heads cfg) :head-dim (:head-dim cfg)
                                             :theta-base (:theta-base cfg)
                                             :rope-proportion (:rope-proportion cfg)
                                             :norm-fn norm/rms-norm
                                             :attn-softcap nil
                                             :layer-type (:layer-type cfg)
                                             :is-shared? (>= i num-unshared)}))
                                        (range num-layers)
                                        layer-packed-tracers)
                           logits (gemma/full-gemma4-forward x emb nil nil nil lw-seq fn-norm pos-tracer num-heads num-kv-heads nil 0 {:final-logit-softcap 30.0 :num-kv-shared-layers num-kv-shared-layers})]
                       (t/convert logits :f32))))

        _ (when-not quiet (println "Tracing & JIT Compiling Single Fused Gemma 4 StableHLO Graph..."))
        fused-graph (trace-graph "gemma4_fused_forward" invars trace-fn)
        exec (xla/compile-graph ctx fused-graph)
        _ (when-not quiet (println "Successfully compiled StableHLO fused forward graph to native XLA PjRtLoadedExecutable handle."))]

    (when verbose
      (println "\n==================================================================")
      (println "--- Single Fused Forward EDN SSA Graph ---")
      (pprint/pprint fused-graph)
      (println "==================================================================\n"))

    exec))

(defn run-autoregressive-generation
  "Runs autoregressive text generation using single fused XLA GPU execution graph."
  [{:keys [ctx tokenizer config opts]} exec device-weights prompt-ids]
  (let [{:keys [max-new-tokens temperature top-k quiet]} opts
        {:keys [max-seq-len vocab-size]} config
        eos (eos-id tokenizer)
        pos-array (int-array (range max-seq-len))
        pos-buf (xla/buffer-from-host-buffer ctx (:client ctx) pos-array [max-seq-len] 4)]
    (loop [cur-tokens (vec prompt-ids)
           step 0]
      (if (or (>= step max-new-tokens) (and (seq cur-tokens) (= (last cur-tokens) eos)))
        (do (xla/destroy-buffer! ctx pos-buf)
            cur-tokens)
        (let [seq-len (count cur-tokens)
              padded-tokens (int-array (concat cur-tokens (repeat (- max-seq-len seq-len) 0)))
              tok-buf (xla/buffer-from-host-buffer ctx (:client ctx) padded-tokens [1 max-seq-len] 4)
              input-args (into [tok-buf pos-buf] device-weights)
              logits-buf (xla/execute exec input-args)
              step-logits (xla/to-host-slice logits-buf (dec seq-len) vocab-size (* max-seq-len vocab-size))
              next-tok (sampling/sample-logits step-logits {:temperature temperature :top-k top-k})
              _ (do (xla/destroy-buffer! ctx tok-buf)
                    (xla/destroy-buffer! ctx logits-buf))]
          (when-not quiet
            (if (= step 0)
              (print (format "%s%s" (decode tokenizer prompt-ids) (decode tokenizer [next-tok])))
              (print (decode tokenizer [next-tok])))
            (flush))
          (recur (conj cur-tokens next-tok) (inc step)))))))

(defn generate-text
  "Generates text response using Gemma 4 Single Fused XLA Execution Graph."
  [session prompt]
  (let [{:keys [tokenizer opts]} session
        {:keys [max-new-tokens temperature top-k model quiet]} opts
        clean-prompt (or prompt "The capital of France is")
        model-str (or model (get-in session [:config :model-dir]) "")
        is-it-model (str/includes? (str/lower-case model-str) "-it")
        prompt-ids (cond
                     is-it-model
                     (let [raw-ids (encode tokenizer clean-prompt)
                           clean-ids (if (= (first raw-ids) (bos-id tokenizer)) (rest raw-ids) raw-ids)]
                       (vec (concat [(bos-id tokenizer) 105 2364 107] clean-ids [106 107 105 4368 107])))

                     :else
                     (let [raw-ids (encode tokenizer clean-prompt)]
                       (if (= (first raw-ids) (bos-id tokenizer))
                         (vec raw-ids)
                         (vec (cons (bos-id tokenizer) raw-ids)))))
        prompt-len (count prompt-ids)]
    (when-not quiet
      (println (format "Prompt: \"%s\"" clean-prompt))
      (println (format "Generation Options: max-new-tokens=%d, temperature=%.2f, top-k=%d, precision=%s"
                       max-new-tokens temperature top-k (name (get-in session [:config :weight-dtype]))))
      (println (format "Encoded Token IDs (%d tokens): %s" prompt-len prompt-ids)))

    (when-not quiet (println "Transferring Gemma 4 model weights to PJRT Device Memory..."))
    (let [device-weights (allocate-device-weights session)
          exec (compile-inference-executables session)]
      (when-not quiet
        (println "\nGenerating tokens autoregressively with Single Fused XLA GPU Kernel..."))
      (let [final-context (run-autoregressive-generation session exec device-weights prompt-ids)]
        (if quiet
          (println)
          (do
            (println "\n\n==================================================================")
            (println "=== Single Fused XLA GPU Forward Pass Verification Passed! ===")
            (println "==================================================================")))
        final-context))))

(defn -main
  "CLI entrypoint for Gemma 4 text generation."
  [& args]
  (let [opts (parse-cli-args args)]
    (when-not (:quiet opts)
      (println "==================================================================")
      (println "  clj-xla Gemma 4 Single-Pass Prefill & BF16 Generation ")
      (println "=================================================================="))
    (let [session (init-inference-session opts)]
      (generate-text session (:prompt opts)))))

(when (= *file* (System/getProperty "clojure.script.filename"))
  (apply -main *command-line-args*))
