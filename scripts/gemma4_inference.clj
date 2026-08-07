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
  [ctx weights-mmap tensor-name shape _weight-dtype weight-enum]
  (if (or (zero? (reduce * 1 shape))
          (not (or (contains? (:header weights-mmap) tensor-name)
                   (contains? (:tensors weights-mmap) tensor-name))))
    (let [num-elements (reduce * 1 shape)
          zero-data (if (= weight-enum 11) (float-array num-elements) (short-array num-elements))]
      (xla/buffer-from-host-buffer ctx (:client ctx) zero-data shape weight-enum))
    (let [slice (st/get-tensor-slice weights-mmap tensor-name)]
      (xla/buffer-from-host-buffer ctx (:client ctx) slice shape weight-enum))))

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
         num-kv-shared-layers (or (:num_kv_shared_layers text-cfg) 20)

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
               :weight-enum weight-enum}})))

(defn allocate-device-weights
  "Transfers all model layer, embedding, and per-layer input weights to PJRT device memory."
  [{:keys [ctx weights-mmap config]}]
  (let [{:keys [prefix-base vocab-size hidden-dim total-pl-dim pl-dim num-layers weight-dtype weight-enum layer-configs]} config
        load-fn (fn [name shape] (load-weight-buffer ctx weights-mmap name shape weight-dtype weight-enum))
        embed-buf (load-fn (str prefix-base "embed_tokens.weight") [vocab-size hidden-dim])
        embed-pl-buf (load-fn (str prefix-base "embed_tokens_per_layer.weight") [vocab-size total-pl-dim])
        pl-model-proj-buf (load-fn (str prefix-base "per_layer_model_projection.weight") [total-pl-dim hidden-dim])
        pl-proj-norm-buf (load-fn (str prefix-base "per_layer_projection_norm.weight") [pl-dim])
        final-norm-buf (load-fn (str prefix-base "norm.weight") [hidden-dim])
        layer-bufs (mapv (fn [i]
                           (let [kmap (gemma/gemma4-weight-key-map i (str prefix-base "layers."))
                                 cfg (nth layer-configs i)
                                 {:keys [q-dim kv-dim head-dim mlp-dim]} cfg]
                             [(load-fn (:input-ln-w kmap) [hidden-dim])
                              (load-fn (:layer-scalar-w kmap) [1])
                              (load-fn (:q-w kmap) [q-dim hidden-dim])
                              (load-fn (:k-w kmap) [kv-dim hidden-dim])
                              (load-fn (:v-w kmap) [kv-dim hidden-dim])
                              (load-fn (:o-w kmap) [hidden-dim q-dim])
                              (load-fn (:q-norm-w kmap) [head-dim])
                              (load-fn (:k-norm-w kmap) [head-dim])
                              (load-fn (:post-attn-ln-w kmap) [hidden-dim])
                              (load-fn (:pre-mlp-ln-w kmap) [hidden-dim])
                              (load-fn (:post-mlp-ln-w kmap) [hidden-dim])
                              (load-fn (:gate-w kmap) [mlp-dim hidden-dim])
                              (load-fn (:up-w kmap) [mlp-dim hidden-dim])
                              (load-fn (:down-w kmap) [hidden-dim mlp-dim])
                              (load-fn (:per-layer-gate-w kmap) [pl-dim hidden-dim])
                              (load-fn (:per-layer-proj-w kmap) [hidden-dim pl-dim])
                              (load-fn (:post-per-layer-norm-w kmap) [hidden-dim])]))
                         (range num-layers))
        flat-layer-bufs (vec (apply concat layer-bufs))]
    (into [embed-buf embed-pl-buf pl-model-proj-buf pl-proj-norm-buf final-norm-buf] flat-layer-bufs)))

(defn allocate-kv-caches
  "Allocates initial zero-filled PJRT device memory buffers for layer K/V caches."
  [{:keys [ctx config]}]
  (let [{:keys [num-layers max-seq-len weight-enum layer-configs]} config]
    (mapv (fn [i]
            (let [cfg (nth layer-configs i)
                  l-nkv (:num-kv-heads cfg)
                  head-dim (:head-dim cfg)
                  c-shape [1 l-nkv max-seq-len head-dim]
                  num-elements (reduce * 1 c-shape)
                  zero-data (if (= weight-enum 11) (float-array num-elements) (short-array num-elements))]
              [(xla/buffer-from-host-buffer ctx (:client ctx) zero-data c-shape weight-enum)
               (xla/buffer-from-host-buffer ctx (:client ctx) zero-data c-shape weight-enum)]))
          (range num-layers))))

(defn compile-inference-executables
  "Traces and JIT-compiles Gemma 4 Single-Pass Prefill and Single-Token Decode StableHLO graphs into PJRT Executables."
  [{:keys [ctx config opts]} prompt-len]
  (let [{:keys [vocab-size hidden-dim total-pl-dim pl-dim max-seq-len num-layers weight-dtype layer-configs num-heads num-kv-heads num-kv-shared-layers]} config
        {:keys [verbose quiet]} opts
        num-unshared (- num-layers num-kv-shared-layers)

        prefill-invars (vec (concat
                             [[:x [:tensor [1 prompt-len] :i32]]
                              [:pos [:tensor [prompt-len] :i32]]
                              [:embed_tokens [:tensor [vocab-size hidden-dim] weight-dtype]]
                              [:embed_tokens_per_layer [:tensor [vocab-size total-pl-dim] weight-dtype]]
                              [:per_layer_model_projection [:tensor [total-pl-dim hidden-dim] weight-dtype]]
                              [:per_layer_projection_norm [:tensor [pl-dim] weight-dtype]]
                              [:final_norm_w [:tensor [hidden-dim] weight-dtype]]]
                             (mapcat (fn [i]
                                       (let [cfg (nth layer-configs i)
                                             {:keys [q-dim kv-dim head-dim mlp-dim]} cfg]
                                         [[(keyword (str "input_ln_w_" i)) [:tensor [hidden-dim] weight-dtype]]
                                          [(keyword (str "layer_scalar_w_" i)) [:tensor [1] weight-dtype]]
                                          [(keyword (str "q_w_" i)) [:tensor [q-dim hidden-dim] weight-dtype]]
                                          [(keyword (str "k_w_" i)) [:tensor [kv-dim hidden-dim] weight-dtype]]
                                          [(keyword (str "v_w_" i)) [:tensor [kv-dim hidden-dim] weight-dtype]]
                                          [(keyword (str "o_w_" i)) [:tensor [hidden-dim q-dim] weight-dtype]]
                                          [(keyword (str "q_norm_w_" i)) [:tensor [head-dim] weight-dtype]]
                                          [(keyword (str "k_norm_w_" i)) [:tensor [head-dim] weight-dtype]]
                                          [(keyword (str "post_attn_ln_w_" i)) [:tensor [hidden-dim] weight-dtype]]
                                          [(keyword (str "pre_mlp_ln_w_" i)) [:tensor [hidden-dim] weight-dtype]]
                                          [(keyword (str "post_mlp_ln_w_" i)) [:tensor [hidden-dim] weight-dtype]]
                                          [(keyword (str "gate_w_" i)) [:tensor [mlp-dim hidden-dim] weight-dtype]]
                                          [(keyword (str "up_w_" i)) [:tensor [mlp-dim hidden-dim] weight-dtype]]
                                          [(keyword (str "down_w_" i)) [:tensor [hidden-dim mlp-dim] weight-dtype]]
                                          [(keyword (str "per_layer_gate_w_" i)) [:tensor [pl-dim hidden-dim] weight-dtype]]
                                          [(keyword (str "per_layer_proj_w_" i)) [:tensor [hidden-dim pl-dim] weight-dtype]]
                                          [(keyword (str "post_per_layer_norm_w_" i)) [:tensor [hidden-dim] weight-dtype]]]))
                                     (range num-layers))
                             (mapcat (fn [i]
                                       (let [cfg (nth layer-configs i)
                                             l-nkv (:num-kv-heads cfg)
                                             head-dim (:head-dim cfg)]
                                         [[(keyword (str "k_cache_" i)) [:tensor [1 l-nkv max-seq-len head-dim] weight-dtype]]
                                          [(keyword (str "v_cache_" i)) [:tensor [1 l-nkv max-seq-len head-dim] weight-dtype]]]))
                                     (range num-layers))))

        decode-invars (assoc prefill-invars
                             0 [:x [:tensor [1 1] :i32]]
                             1 [:pos [:tensor [1] :i32]])

        prefill-trace-fn (fn [x pos-tracer emb emb-pl pl-model-proj pl-proj-norm fn-norm & rest-args]
                           (let [weight-args (take (* 17 num-layers) rest-args)
                                 kv-cache-args (drop (* 17 num-layers) rest-args)
                                 lw-seq (mapv (fn [i [in-ln ls qw kw vw ow qn kn post-attn-ln pre-mlp-ln post-mlp-ln gw uw dw plg plp pln]]
                                                (let [cfg (nth layer-configs i)]
                                                  {:input-ln-w in-ln :layer-scalar-w ls
                                                   :q-w qw :k-w kw :v-w vw :o-w ow
                                                   :q-norm-w qn :k-norm-w kn
                                                   :post-attn-ln-w post-attn-ln :pre-mlp-ln-w pre-mlp-ln :post-mlp-ln-w post-mlp-ln
                                                   :gate-w gw :up-w uw :down-w dw
                                                   :per-layer-gate-w plg :per-layer-proj-w plp :post-per-layer-norm-w pln
                                                   :num-heads num-heads :num-kv-heads num-kv-heads
                                                   :theta-base (:theta-base cfg)
                                                   :rope-proportion (:rope-proportion cfg)
                                                   :norm-fn norm/rms-norm
                                                   :attn-softcap nil
                                                   :layer-type (:layer-type cfg)
                                                   :is-shared? (>= i num-unshared)}))
                                              (range num-layers)
                                              (mapv vec (partition 17 weight-args)))
                                 kv-seq (mapv vec (partition 2 kv-cache-args))
                                 [logits updated-kv-caches] (gemma/full-gemma4-forward x emb emb-pl pl-model-proj pl-proj-norm lw-seq fn-norm pos-tracer num-heads num-kv-heads kv-seq 0 {:final-logit-softcap 30.0 :num-kv-shared-layers num-kv-shared-layers})
                                 f32-logits (t/convert logits :f32)]
                             (into [f32-logits] (apply concat updated-kv-caches))))

        decode-trace-fn (fn [x pos-tracer emb emb-pl pl-model-proj pl-proj-norm fn-norm & rest-args]
                          (let [weight-args (take (* 17 num-layers) rest-args)
                                kv-cache-args (drop (* 17 num-layers) rest-args)
                                lw-seq (mapv (fn [i [in-ln ls qw kw vw ow qn kn post-attn-ln pre-mlp-ln post-mlp-ln gw uw dw plg plp pln]]
                                               (let [cfg (nth layer-configs i)]
                                                 {:input-ln-w in-ln :layer-scalar-w ls
                                                  :q-w qw :k-w kw :v-w vw :o-w ow
                                                  :q-norm-w qn :k-norm-w kn
                                                  :post-attn-ln-w post-attn-ln :pre-mlp-ln-w pre-mlp-ln :post-mlp-ln-w post-mlp-ln
                                                  :gate-w gw :up-w uw :down-w dw
                                                  :per-layer-gate-w plg :per-layer-proj-w plp :post-per-layer-norm-w pln
                                                  :num-heads num-heads :num-kv-heads num-kv-heads
                                                  :theta-base (:theta-base cfg)
                                                  :rope-proportion (:rope-proportion cfg)
                                                  :norm-fn norm/rms-norm
                                                  :attn-softcap nil
                                                  :layer-type (:layer-type cfg)
                                                  :is-shared? (>= i num-unshared)}))
                                             (range num-layers)
                                             (mapv vec (partition 17 weight-args)))
                                kv-seq (mapv vec (partition 2 kv-cache-args))
                                [logits updated-kv-caches] (gemma/full-gemma4-forward x emb emb-pl pl-model-proj pl-proj-norm lw-seq fn-norm pos-tracer num-heads num-kv-heads kv-seq pos-tracer {:final-logit-softcap 30.0 :num-kv-shared-layers num-kv-shared-layers})
                                f32-logits (t/convert logits :f32)]
                            (into [f32-logits] (apply concat updated-kv-caches))))

        _ (when-not quiet (println "Tracing & JIT Compiling Gemma 4 Single-Pass Prefill Graph..."))
        prefill-graph (trace-graph "gemma4_prefill" prefill-invars prefill-trace-fn)
        prefill-exec (xla/compile-graph ctx prefill-graph)

        _ (when-not quiet (println "Tracing & JIT Compiling Gemma 4 Single-Token Decoding Graph..."))
        decode-graph (trace-graph "gemma4_decode" decode-invars decode-trace-fn)
        decode-exec (xla/compile-graph ctx decode-graph)
        _ (when-not quiet (println "Successfully compiled StableHLO prefill and decode graphs to native XLA PjRtLoadedExecutable handles."))]

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
  [{:keys [ctx tokenizer config opts]} executables device-weights initial-kv-bufs prompt-ids]
  (let [prompt-len (count prompt-ids)
        {:keys [vocab-size]} config
        {:keys [temperature top-k quiet]} opts
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
        indexed (map-indexed vector (vec last-logits))
        top-10 (take 10 (sort-by second > indexed))
        _ (when-not quiet
            (println "\n=== Top 10 Prefill Predicted Tokens ===")
            (doseq [[id logit] top-10]
              (println (format "  token %6d | logit: %8.4f | text: %s" id logit (pr-str (decode tokenizer [id]))))))
        first-gen-tok (sampling/sample-logits last-logits {:temperature temperature :top-k top-k})]
    {:first-gen-tok first-gen-tok
     :prefill-kv-bufs prefill-kv-bufs
     :prompt-ids prompt-ids}))

(defn run-autoregressive-decode
  "Runs the single-token autoregressive decoding loop."
  [{:keys [ctx tokenizer config opts]} executables device-weights prefill-result]
  (let [{:keys [max-new-tokens temperature top-k quiet]} opts
        {:keys [vocab-size]} config
        {:keys [decode-exec]} executables
        {:keys [first-gen-tok prefill-kv-bufs prompt-ids]} prefill-result
        prompt-len (count prompt-ids)
        eos (eos-id tokenizer)]
    (print (decode tokenizer [first-gen-tok]))
    (flush)
    (if (= first-gen-tok eos)
      (do (when-not quiet (println "\nReached EOS token."))
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
            (if (or (= next-tok eos) (= next-tok 106))
              next-context
              (recur next-context updated-kv-bufs (inc pos) (inc step)))))))))

(defn generate-text
  "Top-level REPL/programmatic helper: runs full end-to-end text generation on an initialized session."
  ([session] (generate-text session (or (:prompt (:opts session)) "The capital of France is")))
  ([session prompt-text]
   (let [{:keys [tokenizer opts]} session
         {:keys [max-new-tokens temperature top-k quiet]} opts
         clean-prompt (str/replace prompt-text #"\\n" "\n")
         model-dir (get-in session [:config :model-dir] "")
         is-it-model? (or (str/includes? model-dir "-it") (str/includes? clean-prompt "<|turn>"))
         prompt-ids (cond
                      (str/includes? clean-prompt "<|turn>")
                      ;; Parse raw control string into special token IDs
                      (let [parts (clojure.string/split clean-prompt #"(?=<\|turn>)|(?<=<\|turn>)|(?=<turn\|>)|(?<=<turn\|>)")
                            toks (mapcat (fn [p]
                                           (cond
                                             (= p "<|turn>") [105]
                                             (= p "<turn|>") [106]
                                             :else (encode tokenizer p)))
                                         parts)]
                        (into [(bos-id tokenizer)] toks))

                      is-it-model?
                      ;; Wrap plain question/prompt into Gemma 4 IT Turn Template
                      (let [raw-ids (encode tokenizer clean-prompt)
                            clean-ids (if (= (first raw-ids) (bos-id tokenizer)) (rest raw-ids) raw-ids)]
                        (vec (concat [(bos-id tokenizer) 105 2364 107] clean-ids [106 107 105 4368 107])))

                      :else
                      ;; Base Pretrained model (e.g. google/gemma-4-E2B): direct prompt completion
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
           initial-kv-bufs (allocate-kv-caches session)
           executables (compile-inference-executables session prompt-len)]
       (when-not quiet
         (println "\nGenerating tokens autoregressively with Gemma 4 Single-Pass Prefill...")
         (print clean-prompt)
         (flush))
       (let [prefill-res (run-prompt-prefill session executables device-weights initial-kv-bufs prompt-ids)
             final-context (run-autoregressive-decode session executables device-weights prefill-res)]
         (if quiet
           (println)
           (do
             (println "\n\n==================================================================")
             (println "=== End-to-End Gemma 4 Single-Pass Prefill Verification Passed! ===")
             (println "==================================================================")))
         final-context)))))

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
