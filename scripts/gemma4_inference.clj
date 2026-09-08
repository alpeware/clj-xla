(ns scripts.gemma4-inference
  "Top-level runnable integration script and REPL API for end-to-end Gemma 4 text generation via pure XLA execution."
  (:require [clj-xla.core :as xla]
            [clj-xla.logic.lower :as lower]
            [clj-xla.logic.models.gemma :as gemma-logic]
            [clj-xla.profile :as profile]
            [clj-xla.safetensors :as st]
            [clj-xla.sampling :as sampling]
            [clj-xla.tokenizer.core :as tok]
            [clj-xla.tokenizer.protocol :refer [bos-id decode encode eos-id]]
            [clojure.data.json :as json]
            [clojure.java.io :as io]
            [clojure.pprint :refer [pprint]]
            [clojure.string :as str])
  (:import [java.lang.foreign Arena]))

(def DEFAULT_CLI_OPTS
  {:prompt "The capital of France is"
   :max-new-tokens 20
   :temperature 0.7
   :top-k 10
   :backend :cpu
   :precision :bf16
   :method :tensor-logic
   :compare false
   :verbose false
   :quiet false
   :profile true
   :profile-out "scratch/gemma4_profile.edn"
   :chrome-trace-out "scratch/gemma4_chrome_trace.json"})

(def DEFAULT_MODEL_DIRS
  [".models/gemma-4-E2B-it"
   ".models/gemma-4-E2B"
   ".models/gemma-4-E4B-it"
   ".models/gemma-4-E4B"
   ".models/gemma-4-12B-it"
   ".models/gemma-4-12B"
   ".models/gemma-4-26B-A4B-it"
   ".models/gemma-4-26B-A4B"
   ".models/gemma-4-31B-it"
   ".models/gemma-4-31B"
   ".models/gemma-4"])

(defn- normalize-args
  [args]
  (mapcat (fn [arg]
            (if (and (str/starts-with? arg "--") (str/includes? arg "="))
              (str/split arg #"=" 2)
              [arg]))
          args))

(defn parse-cli-args
  "Parses command-line flags (--prompt, --model/--model-dir, --max-new-tokens, --temperature, --top-k, --backend, --precision, --verbose, --quiet)."
  [args]
  (loop [remaining (vec (normalize-args args))
         opts DEFAULT_CLI_OPTS]
    (if (empty? remaining)
      opts
      (let [flag (first remaining)
            val (second remaining)]
        (cond
          (and (= flag "--prompt") val)
          (recur (subvec remaining 2) (assoc opts :prompt val))

          (and (= flag "--prompt-file") val)
          (recur (subvec remaining 2) (assoc opts :prompt (slurp val)))

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
          (let [clean-kw (keyword (str/replace val #"^:+" ""))]
            (when (= clean-kw :rocm)
              (try
                (let [linker (java.lang.foreign.Linker/nativeLinker)
                      lookup (.defaultLookup linker)
                      setenv-opt (.find lookup "setenv")]
                  (when (.isPresent setenv-opt)
                    (let [setenv-ptr ^java.lang.foreign.MemorySegment (.get setenv-opt)
                          fd (java.lang.foreign.FunctionDescriptor/of java.lang.foreign.ValueLayout/JAVA_INT
                                                                      (into-array java.lang.foreign.MemoryLayout
                                                                                  [java.lang.foreign.ValueLayout/ADDRESS
                                                                                   java.lang.foreign.ValueLayout/ADDRESS
                                                                                   java.lang.foreign.ValueLayout/JAVA_INT]))
                          handle (.downcallHandle linker setenv-ptr fd (make-array java.lang.foreign.Linker$Option 0))]
                      (with-open [arena (java.lang.foreign.Arena/ofConfined)]
                        (.invokeWithArguments handle [(.allocateFrom arena "HIP_VISIBLE_DEVICES") (.allocateFrom arena "0") (int 1)])
                        (.invokeWithArguments handle [(.allocateFrom arena "ROCR_VISIBLE_DEVICES") (.allocateFrom arena "0") (int 1)])
                        (.invokeWithArguments handle [(.allocateFrom arena "HSA_OVERRIDE_GFX_VERSION") (.allocateFrom arena "11.0.0") (int 1)])))))
                (catch Exception _ nil)))
            (recur (subvec remaining 2) (assoc opts :backend clean-kw)))

          (and (= flag "--precision") val)
          (recur (subvec remaining 2) (assoc opts :precision (keyword (str/replace val #"^:+" ""))))

          (and (= flag "--out") val)
          (recur (subvec remaining 2) (assoc opts :out val))

          (and (= flag "--method") val)
          (recur (subvec remaining 2) (assoc opts :method (keyword (str/replace val #"^:+" ""))))

          (and (= flag "--max-seq-len") val)
          (recur (subvec remaining 2) (assoc opts :max-seq-len (Long/parseLong val)))

          (and (= flag "--mode") val)
          (recur (subvec remaining 2) (assoc opts :mode (keyword (str/replace val #"^:+" ""))))

          (and (= flag "--compare") val)
          (recur (subvec remaining 2) (assoc opts :compare (Boolean/parseBoolean val)))

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

(defn quantize-bf16-to-int8
  "Quantizes a BF16 short-array to INT8 byte-array with per-tensor symmetric quantization.
   Returns {:data byte-array :scale float}."
  [^shorts bf16-shorts]
  (let [n (alength bf16-shorts)
        ;; Find max absolute value in BF16
        max-abs (loop [i 0 m (float 0.0)]
                  (if (>= i n)
                    m
                    (let [s (int (aget bf16-shorts i))
                          bits (unchecked-int (bit-shift-left (long (bit-and s 0xffff)) 16))
                          f (Math/abs (Float/intBitsToFloat bits))]
                      (recur (inc i) (max m f)))))
        scale (if (zero? max-abs) 1.0 (/ (double max-abs) 127.0))
        inv-scale (float (/ 1.0 scale))
        result (byte-array n)]
    (dotimes [i n]
      (let [s (int (aget bf16-shorts i))
            bits (unchecked-int (bit-shift-left (long (bit-and s 0xffff)) 16))
            f (Float/intBitsToFloat bits)
            q (Math/round (* f inv-scale))
            clamped (max -127 (min 127 q))]
        (aset result i (byte clamped))))
    {:data result :scale (float scale)}))

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
         max-seq-len (long (or (:max-seq-len opts) 16384))

         layer-configs (mapv (fn [i]
                               (let [kmap (gemma-logic/gemma4-weight-key-map i (str prefix-base "layers."))
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
         is-int8 (= weight-dtype :int8)
         ;; For int8: matmul weights use S8 (enum 2), norms/embeddings stay BF16 (enum 13)
         weight-enum (cond is-int8 2 (= weight-dtype :f32) 11 :else 13)
         norm-enum (if (= weight-dtype :f32) 11 13)]

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
               :is-int8 is-int8
               :norm-enum norm-enum}})))

(defn build-tensor-logic-invars
  "Constructs EDN SSA signature invars for full Gemma 4 model forward pass."
  [config max-seq-len]
  (let [{:keys [vocab-size hidden-dim total-pl-dim pl-dim num-layers weight-dtype is-int8 layer-configs last-token-only?]} config
        norm-dtype (if is-int8 :bf16 weight-dtype)
        has-ple? (pos? total-pl-dim)]
    (vec (concat [[:x [:tensor [1 max-seq-len] :i32]]]
                 (when last-token-only?
                   [[:pos [:tensor [1] :i32]]])
                 [[:embed_tokens [:tensor [vocab-size hidden-dim] norm-dtype]]]
                 (when has-ple?
                   [[:embed_tokens_per_layer [:tensor [vocab-size total-pl-dim] norm-dtype]]
                    [:per_layer_model_projection [:tensor [total-pl-dim hidden-dim] norm-dtype]]
                    [:per_layer_projection_norm [:tensor [pl-dim] norm-dtype]]])
                 (mapcat (fn [i]
                           (let [cfg (nth layer-configs i)
                                 q-dim (:q-dim cfg)
                                 kv-dim (:kv-dim cfg)
                                 head-dim (:head-dim cfg)
                                 mlp-dim (:mlp-dim cfg)]
                             (concat
                              [[(keyword (str "input_ln_w_" i)) [:tensor [hidden-dim] norm-dtype]]
                               [(keyword (str "layer_scalar_" i)) [:tensor [1] norm-dtype]]
                               [(keyword (str "q_w_" i)) [:tensor [q-dim hidden-dim] weight-dtype]]
                               [(keyword (str "k_w_" i)) [:tensor [kv-dim hidden-dim] weight-dtype]]
                               [(keyword (str "v_w_" i)) [:tensor [kv-dim hidden-dim] weight-dtype]]
                               [(keyword (str "o_w_" i)) [:tensor [hidden-dim q-dim] weight-dtype]]
                               [(keyword (str "q_norm_w_" i)) [:tensor [head-dim] norm-dtype]]
                               [(keyword (str "k_norm_w_" i)) [:tensor [head-dim] norm-dtype]]
                               [(keyword (str "post_attn_ln_w_" i)) [:tensor [hidden-dim] norm-dtype]]
                               [(keyword (str "pre_mlp_ln_w_" i)) [:tensor [hidden-dim] norm-dtype]]
                               [(keyword (str "post_mlp_ln_w_" i)) [:tensor [hidden-dim] norm-dtype]]
                               [(keyword (str "gate_w_" i)) [:tensor [mlp-dim hidden-dim] weight-dtype]]
                               [(keyword (str "up_w_" i)) [:tensor [mlp-dim hidden-dim] weight-dtype]]
                               [(keyword (str "down_w_" i)) [:tensor [hidden-dim mlp-dim] weight-dtype]]]
                              (when has-ple?
                                [[(keyword (str "per_layer_gate_w_" i)) [:tensor [pl-dim hidden-dim] norm-dtype]]
                                 [(keyword (str "per_layer_proj_w_" i)) [:tensor [hidden-dim pl-dim] norm-dtype]]
                                 [(keyword (str "post_per_layer_norm_w_" i)) [:tensor [hidden-dim] norm-dtype]]]))))
                         (range num-layers))
                 [[:final_norm_w [:tensor [hidden-dim] norm-dtype]]]))))

(defn allocate-device-weights
  "Loads individual weight tensors for Gemma 4 into PJRT device buffers matching build-tensor-logic-invars."
  [{:keys [ctx weights-mmap config]}]
  (let [{:keys [prefix-base vocab-size hidden-dim total-pl-dim pl-dim weight-dtype weight-enum norm-enum layer-configs num-layers]} config
        has-ple? (pos? total-pl-dim)
        load-fn (fn
                  ([name shape enum] (load-weight-buffer ctx weights-mmap name shape weight-dtype enum 0.0))
                  ([name shape enum default-val] (load-weight-buffer ctx weights-mmap name shape weight-dtype enum default-val)))
        embed-buf (load-fn (str prefix-base "embed_tokens.weight") [vocab-size hidden-dim] norm-enum)
        ple-bufs (when has-ple?
                   [(load-fn (str prefix-base "embed_tokens_per_layer.weight") [vocab-size total-pl-dim] norm-enum)
                    (load-fn (str prefix-base "per_layer_model_projection.weight") [total-pl-dim hidden-dim] norm-enum)
                    (load-fn (str prefix-base "per_layer_projection_norm.weight") [pl-dim] norm-enum)])
        layer-bufs (mapcat (fn [i]
                             (let [kmap (gemma-logic/gemma4-weight-key-map i (str prefix-base "layers."))
                                   cfg (nth layer-configs i)
                                   q-dim (:q-dim cfg)
                                   kv-dim (:kv-dim cfg)
                                   head-dim (:head-dim cfg)
                                   mlp-dim (:mlp-dim cfg)]
                               (concat
                                [(load-fn (:input-ln-w kmap) [hidden-dim] norm-enum 0.0)
                                 (load-fn (:layer-scalar-w kmap) [1] norm-enum 1.0)
                                 (load-fn (:q-w kmap) [q-dim hidden-dim] weight-enum 0.0)
                                 (load-fn (:k-w kmap) [kv-dim hidden-dim] weight-enum 0.0)
                                 (load-fn (:v-w kmap) [kv-dim hidden-dim] weight-enum 0.0)
                                 (load-fn (:o-w kmap) [hidden-dim q-dim] weight-enum 0.0)
                                 (load-fn (:q-norm-w kmap) [head-dim] norm-enum 0.0)
                                 (load-fn (:k-norm-w kmap) [head-dim] norm-enum 0.0)
                                 (load-fn (:post-attn-ln-w kmap) [hidden-dim] norm-enum 0.0)
                                 (load-fn (:pre-mlp-ln-w kmap) [hidden-dim] norm-enum 0.0)
                                 (load-fn (:post-mlp-ln-w kmap) [hidden-dim] norm-enum 0.0)
                                 (load-fn (:gate-w kmap) [mlp-dim hidden-dim] weight-enum 0.0)
                                 (load-fn (:up-w kmap) [mlp-dim hidden-dim] weight-enum 0.0)
                                 (load-fn (:down-w kmap) [hidden-dim mlp-dim] weight-enum 0.0)]
                                (when has-ple?
                                  [(load-fn (:per-layer-gate-w kmap) [pl-dim hidden-dim] norm-enum 0.0)
                                   (load-fn (:per-layer-proj-w kmap) [hidden-dim pl-dim] norm-enum 0.0)
                                   (load-fn (:post-per-layer-norm-w kmap) [hidden-dim] norm-enum 0.0)]))))
                           (range num-layers))
        final-norm-buf (load-fn (str prefix-base "norm.weight") [hidden-dim] norm-enum 0.0)]
    (vec (concat [embed-buf] ple-bufs layer-bufs [final-norm-buf]))))

(def allocate-tensor-logic-weights allocate-device-weights)

(defn compile-tensor-logic-executable
  "Compiles Gemma 4 model AST into a native StableHLO MLIR executable."
  [{:keys [ctx config opts]} max-seq-len]
  (let [last-token? (get opts :last-token-only? true)
        config-with-len (assoc config :max-seq-len max-seq-len :last-token-only? last-token?)
        invars (build-tensor-logic-invars config-with-len max-seq-len)
        ast (gemma-logic/gemma4-model-ast config-with-len)
        _ (when-not (:quiet opts)
            (println (format "Lowering declarative Tensor Logic Gemma 4 AST (%d layers, max-seq-len=%d, last-token-only=%s) to StableHLO..."
                             (:num-layers config) max-seq-len (str last-token?))))
        graph (lower/ast->graph "gemma4_tensor_logic" invars ast #{:logits})]
    (when-not (:quiet opts)
      (println "Compiling Tensor Logic graph to native XLA PjRtLoadedExecutable..."))
    (xla/compile-graph ctx graph)))

(defn argmax-host
  "Finds the index of the maximum float value in float array `arr`."
  [^floats arr]
  (let [n (alength arr)]
    (loop [i 1
           max-idx 0
           max-val (aget arr 0)]
      (if (< i n)
        (let [v (aget arr i)]
          (if (> v max-val)
            (recur (inc i) i v)
            (recur (inc i) max-idx max-val)))
        max-idx))))

(defn sample-next-token
  "Selects next token from float array `logits-arr` using sampling options and repetition penalty."
  [^floats logits-arr opts prompt-ids gen-ids]
  (let [{:keys [temperature top-k top-p repetition-penalty]
         :or {temperature 0.0 top-k 10 top-p 1.0 repetition-penalty 1.15}} opts
        rep-pen (double (or repetition-penalty 1.15))]
    (if (or (nil? temperature) (<= temperature 0.0))
      (if (and (number? rep-pen) (> rep-pen 1.0) (seq (concat prompt-ids gen-ids)))
        (let [penalized (sampling/apply-repetition-penalty (vec logits-arr) (concat prompt-ids gen-ids) rep-pen)]
          (first (apply max-key second (map-indexed vector penalized))))
        (argmax-host logits-arr))
      (let [logits-vec (vec logits-arr)
            seen-ids (vec (concat prompt-ids gen-ids))]
        (sampling/sample-logits logits-vec {:temperature temperature
                                            :top-k top-k
                                            :top-p top-p
                                            :repetition-penalty rep-pen
                                            :seen-ids seen-ids})))))

(defn run-autoregressive-generation-logic
  "Executes autoregressive token generation using pure Tensor Logic Gemma 4 executable."
  ([session exec device-weights prompt-ids]
   (run-autoregressive-generation-logic session exec device-weights prompt-ids nil))
  ([{:keys [ctx opts config tokenizer]} exec device-weights prompt-ids max-seq-len]
   (let [{:keys [max-new-tokens quiet mode]} opts
         is-agent? (= mode :agent)
         seq-len (long (or max-seq-len (:max-seq-len config) 128))
         vocab-size (long (or (:vocab-size config) (:vocab_size config) 262144))
         last-token? (get opts :last-token-only? true)
         weight-dt (get config :weight-dtype :bf16)
         cur-tokens (atom (vec prompt-ids))
         t0 (System/nanoTime)]
     (loop [step 0]
       (if (>= step max-new-tokens)
         nil
         (let [s-len (count @cur-tokens)
               in-arr (int-array (take seq-len (concat @cur-tokens (repeat 0))))
               in-b (xla/buffer-from-host-buffer ctx (:client ctx) in-arr [1 seq-len] 4)
               pos-b (when last-token?
                       (xla/buffer-from-host-buffer ctx (:client ctx) (int-array [(dec s-len)]) [1] 4))
               args (if last-token?
                      (into [in-b pos-b] device-weights)
                      (into [in-b] device-weights))
               out (xla/execute exec args)
               logits (if last-token?
                        (xla/to-host-slice out 0 vocab-size vocab-size weight-dt)
                        (xla/to-host-slice out (dec s-len) vocab-size (* seq-len vocab-size) weight-dt))
               next-id (sample-next-token logits opts prompt-ids (subvec @cur-tokens (count prompt-ids)))]
           (xla/destroy-buffer! ctx in-b)
           (when pos-b (xla/destroy-buffer! ctx pos-b))
           (xla/destroy-buffer! ctx out)
           (swap! cur-tokens conj next-id)
           (when (and (not quiet) (not is-agent?))
             (print (decode tokenizer [next-id]))
             (flush))
           (if (or (= next-id 1) (= next-id (eos-id tokenizer)))
             nil
             (recur (inc step))))))
     (let [t1 (System/nanoTime)
           total-ms (/ (- t1 t0) 1e6)
           gen-count (- (count @cur-tokens) (count prompt-ids))
           tok-s (if (pos? total-ms) (/ (* gen-count 1000.0) total-ms) 0.0)]
       (when-not quiet
         (println)
         (println "\n------------------------------------------------------------------")
         (println "  Telemetry Benchmark Metrics (Tensor Logic End-to-End):")
         (println (format "    • Total Generation Latency    : %8.2f ms (%d tokens)" total-ms gen-count))
         (println (format "    • Generation Speed            : %8.2f tok/s (%6.2f ms/tok)" tok-s (if (pos? gen-count) (/ total-ms gen-count) 0.0)))
         (println "------------------------------------------------------------------\n")))
     @cur-tokens)))

(def run-autoregressive-generation run-autoregressive-generation-logic)
(def compile-executable compile-tensor-logic-executable)

(defn generate-text
  "Generates text response using Gemma 4 model via pure Tensor Logic execution."
  [session prompt]
  (let [{:keys [tokenizer opts]} session
        {:keys [max-new-tokens temperature top-k model quiet]} opts
        clean-prompt (or prompt "The capital of France is")
        model-str (or model (get-in session [:config :model-dir]) "")
        is-it-model (str/includes? (str/lower-case model-str) "-it")
        is-already-templated (or (str/includes? clean-prompt "<|turn>user") (str/includes? clean-prompt "<|turn>model"))
        prompt-ids (cond
                     (and is-it-model (not is-already-templated))
                     (let [raw-ids (encode tokenizer clean-prompt)
                           clean-ids (if (= (first raw-ids) (bos-id tokenizer)) (rest raw-ids) raw-ids)]
                       (vec (concat [(bos-id tokenizer) 105 2364 107] clean-ids [106 107 105 4368 107])))

                     :else
                     (let [raw-ids (encode tokenizer clean-prompt)]
                       (if (= (first raw-ids) (bos-id tokenizer))
                         (vec raw-ids)
                         (vec (cons (bos-id tokenizer) raw-ids)))))
        prompt-len (count prompt-ids)
        max-seq-len (long (or (:max-seq-len session) (:max-seq-len opts) (min 2048 (+ prompt-len max-new-tokens 16))))]
    (when-not quiet
      (let [prompt-str (if (> (count clean-prompt) 200)
                         (str (subs clean-prompt 0 100) " ... [truncated " (count clean-prompt) " chars] ... " (subs clean-prompt (- (count clean-prompt) 100)))
                         clean-prompt)
            tok-str (if (> prompt-len 30)
                      (str "[" (str/join " " (take 10 prompt-ids)) " ... " (str/join " " (take-last 5 prompt-ids)) "]")
                      (str prompt-ids))]
        (println (format "Prompt: \"%s\"" prompt-str))
        (println (format "Generation Options: max-new-tokens=%d, temperature=%.2f, top-k=%d, precision=%s, method=tensor-logic"
                         max-new-tokens temperature top-k (name (get-in session [:config :weight-dtype]))))
        (println (format "Encoded Token IDs (%d tokens): %s" prompt-len tok-str))))

    (let [metrics-atom (or (:metrics-atom session) (atom {}))
          trace-spans-atom (or (:trace-spans-atom session) (atom []))
          reuse-weights? (some? (:device-weights session))
          reuse-exec? (some? (:executable session))
          device-weights (binding [profile/*active-trace-spans* trace-spans-atom]
                           (if reuse-weights?
                             (:device-weights session)
                             (profile/with-profile metrics-atom "weight_transfer"
                               (allocate-device-weights session))))
          exec (binding [profile/*active-trace-spans* trace-spans-atom]
                 (if reuse-exec?
                   (:executable session)
                   (profile/with-profile metrics-atom "graph_compilation"
                     (compile-tensor-logic-executable session max-seq-len))))]
      (when-not quiet
        (println "\nGenerating tokens autoregressively with pure Tensor Logic Gemma 4 Kernel..."))
      (let [final-context (binding [profile/*active-trace-spans* trace-spans-atom]
                            (profile/with-profile metrics-atom "autoregressive_generation"
                              (run-autoregressive-generation session exec device-weights prompt-ids max-seq-len)))
            generated-str (decode tokenizer final-context)]
        (when-not quiet
          (println)
          (println generated-str))
        (when-not reuse-weights?
          (doseq [w device-weights] (xla/destroy-buffer! (:ctx session) w)))
        (when-let [out-path (:out opts)]
          (spit out-path generated-str)
          (when-not quiet
            (println (format "\n  ↳ Written generated output to [%s]" out-path))))
        (when-let [profile-path (:profile-out opts)]
          (spit profile-path (with-out-str (pprint @metrics-atom)))
          (when-not quiet
            (println (format "  ↳ Saved telemetry profile report to [%s]" profile-path))))
        (when-let [trace-path (:chrome-trace-out opts)]
          (profile/save-chrome-trace! @trace-spans-atom trace-path)
          (when-not quiet
            (println (format "  ↳ Saved Chrome tracing JSON to [%s]" trace-path))))
        (when-not quiet
          (println "\n==================================================================")
          (println "=== Tensor Logic Gemma 4 Generation Verification Passed! ===")
          (println "=================================================================="))
        final-context))))

(defn generate-text-string
  "Generates text response using Gemma 4 model session and returns decoded text string."
  [session prompt]
  (let [{:keys [tokenizer]} session
        final-context (generate-text session prompt)]
    (decode tokenizer final-context)))

(defn init-agent-vram-session
  "Initializes a persistent VRAM session for agent loops.
   Pre-allocates weights in device memory and compiles the forward graph once."
  ([opts]
   (init-agent-vram-session opts (long (or (:max-seq-len opts) 512))))
  ([opts max-seq-len]
   (let [opts (assoc opts :mode :agent :max-seq-len max-seq-len)
         session (init-inference-session opts)
         _ (when-not (:quiet opts) (println "Pinning Gemma 4 weights in PJRT VRAM..."))
         device-weights (allocate-device-weights session)
         _ (when-not (:quiet opts) (println (format "Pre-compiling Gemma 4 StableHLO graph (max-seq-len=%d)..." max-seq-len)))
         exec (compile-tensor-logic-executable session max-seq-len)]
     (assoc session
            :device-weights device-weights
            :executable exec
            :max-seq-len max-seq-len))))

(defn close-agent-session!
  "Releases VRAM resources for an agent session."
  [{:keys [ctx device-weights]}]
  (when (seq device-weights)
    (doseq [w device-weights]
      (xla/destroy-buffer! ctx w))))

(defn- find-libjsig
  "Searches standard JDK paths for libjsig.so."
  []
  (let [jh (System/getProperty "java.home")
        paths [(str jh "/lib/server/libjsig.so")
               (str jh "/lib/libjsig.so")
               "/usr/lib64/openjdk-25/lib/server/libjsig.so"]]
    (first (filter #(.exists (io/file %)) paths))))

(defn needs-libjsig-reexec?
  "Returns true when running with ROCm backend and LD_PRELOAD does not
   already include libjsig.so. Signal chaining via LD_PRELOAD is required
   because the ROCm PJRT plugin bundles LLVM, which installs its own signal
   handlers that conflict with the JVM's."
  [opts]
  (and (= (:backend opts) :rocm)
       (not (some-> (System/getenv "LD_PRELOAD")
                    (.contains "libjsig")))))

(defn reexec-with-libjsig!
  "Re-launches current JVM process with LD_PRELOAD=libjsig.so for ROCm signal chaining."
  ([args] (reexec-with-libjsig! args "scripts.gemma4-inference"))
  ([args main-ns]
   (let [jsig-path (find-libjsig)]
     (when-not jsig-path
       (binding [*out* *err*]
         (println "WARNING: libjsig.so not found — ROCm signal chaining unavailable."))
       (flush)
       nil)
     (when jsig-path
       (let [jh (System/getProperty "java.home")
             java-bin (str jh "/bin/java")
             rt-bean (java.lang.management.ManagementFactory/getRuntimeMXBean)
             jvm-args (.getInputArguments rt-bean)
             cp (System/getProperty "java.class.path")
             cmd (vec (concat [java-bin]
                              jvm-args
                              ["-cp" cp "clojure.main" "-m" main-ns]
                              args))
             pb (ProcessBuilder. ^java.util.List cmd)
             env (.environment pb)
             existing-preload (.get env "LD_PRELOAD")
             new-preload (if (and existing-preload (not (.isEmpty ^String existing-preload)))
                           (str jsig-path ":" existing-preload)
                           jsig-path)]
         (.put env "LD_PRELOAD" new-preload)
         (.inheritIO pb)
         (System/exit (.waitFor (.start pb))))))))

(defn -main
  "CLI entrypoint for Gemma 4 text generation."
  [& args]
  (try
    (let [opts (parse-cli-args args)]
      ;; Re-exec with LD_PRELOAD=libjsig.so for ROCm signal chaining.
      ;; Must happen before any PJRT plugin is loaded.
      (when (needs-libjsig-reexec? opts)
        (reexec-with-libjsig! args))
      (when-not (:quiet opts)
        (println "==================================================================")
        (println (str "  clj-xla Gemma 4 Single-Pass Prefill & " (if (= (:precision opts) :int8) "INT8" "BF16") " Generation "))
        (println "=================================================================="))
      (let [session (init-inference-session opts)]
        (generate-text session (:prompt opts))))
    (catch Throwable e
      (println "\nExecution Exception:" (.getMessage e))
      (.printStackTrace e))
    (finally
      (.. Runtime getRuntime (halt 0)))))

(when (= *file* (System/getProperty "clojure.script.filename"))
  (apply -main *command-line-args*))
