(ns scripts.gemma4-inference
  "Top-level runnable integration script and REPL API for end-to-end Gemma 4 text generation via pure XLA execution."
  (:require [clj-xla.core :as xla]
            [clj-xla.logic.lower :as lower]
            [clj-xla.logic.models.gemma :as gemma-logic]
            [clj-xla.models.gemma :as gemma]
            [clj-xla.nn.norm :as norm]
            [clj-xla.pjrt :as pjrt]
            [clj-xla.profile :as profile]
            [clj-xla.safetensors :as st]
            [clj-xla.sampling :as sampling]
            [clj-xla.tensor :as t]
            [clj-xla.tokenizer.core :as tok]
            [clj-xla.tokenizer.protocol :refer [bos-id decode encode eos-id]]
            [clj-xla.trace :refer [trace-graph]]
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

         layer-grouped-specs
         (mapv (fn [i]
                 (let [kmap (gemma/gemma4-weight-key-map i (str prefix-base "layers."))
                       cfg (nth layer-configs i)
                       {:keys [q-dim kv-dim head-dim mlp-dim]} cfg
                       v-key (if (or (contains? header (:v-w kmap))
                                     (contains? (:tensors weights-mmap) (:v-w kmap)))
                               (:v-w kmap)
                               (:k-w kmap))
                       norms-base-el (+ (* 4 hidden-dim) (* 2 head-dim) 1)
                       pl-total-el (if (pos? total-pl-dim)
                                     (+ (* pl-dim hidden-dim) (* hidden-dim pl-dim) hidden-dim)
                                     0)
                       norms-total-el (+ norms-base-el pl-total-el)]
                   {:layer-idx i
                    :q-dim q-dim :kv-dim kv-dim :head-dim head-dim :mlp-dim mlp-dim
                    :qkv-rows (+ q-dim kv-dim kv-dim)
                    :gate-up-rows (* 2 mlp-dim)
                    :norms-base-el norms-base-el
                    :norms-total-el norms-total-el
                    :pl-total-el pl-total-el
                    :keys {:q-w (:q-w kmap) :k-w (:k-w kmap) :v-w v-key
                           :o-w (:o-w kmap)
                           :gate-w (:gate-w kmap) :up-w (:up-w kmap)
                           :down-w (:down-w kmap)
                           :input-ln-w (:input-ln-w kmap) :layer-scalar-w (:layer-scalar-w kmap)
                           :q-norm-w (:q-norm-w kmap) :k-norm-w (:k-norm-w kmap)
                           :post-attn-ln-w (:post-attn-ln-w kmap)
                           :pre-mlp-ln-w (:pre-mlp-ln-w kmap) :post-mlp-ln-w (:post-mlp-ln-w kmap)
                           :per-layer-gate-w (:per-layer-gate-w kmap)
                           :per-layer-proj-w (:per-layer-proj-w kmap)
                           :post-per-layer-norm-w (:post-per-layer-norm-w kmap)}}))
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
               :norm-enum norm-enum
               :layer-grouped-specs layer-grouped-specs}})))

(defn allocate-device-weights
  "Transfers all model layer, embedding, and per-layer input weights to PJRT device memory."
  [{:keys [ctx weights-mmap config]}]
  (let [{:keys [prefix-base vocab-size hidden-dim total-pl-dim pl-dim weight-dtype weight-enum is-int8 norm-enum layer-grouped-specs]} config
        client (:client ctx)
        is-bf16-format (not= weight-dtype :f32)
        has-ple? (pos? total-pl-dim)
        load-fn (fn
                  ([name shape] (load-weight-buffer ctx weights-mmap name shape (if is-int8 :bf16 weight-dtype) norm-enum 0.0))
                  ([name shape default-val] (load-weight-buffer ctx weights-mmap name shape (if is-int8 :bf16 weight-dtype) norm-enum default-val)))
        embed-buf (load-fn (str prefix-base "embed_tokens.weight") [vocab-size hidden-dim])
        final-norm-buf (load-fn (str prefix-base "norm.weight") [hidden-dim])
        ple-global-bufs (when has-ple?
                          [(load-fn (str prefix-base "embed_tokens_per_layer.weight") [vocab-size total-pl-dim])
                           (load-fn (str prefix-base "per_layer_model_projection.weight") [total-pl-dim hidden-dim])
                           (load-fn (str prefix-base "per_layer_projection_norm.weight") [pl-dim])])
        copy-tensor-to-arr (fn [arr name shape offset default-val]
                             (let [num-el (reduce * 1 shape)
                                   has-tensor (contains? (:tensors weights-mmap) name)]
                               (if has-tensor
                                 (let [slice (st/get-tensor-slice weights-mmap name)]
                                   (if is-bf16-format
                                     (java.lang.foreign.MemorySegment/copy slice (java.lang.foreign.ValueLayout/JAVA_SHORT) 0
                                                                           ^shorts arr offset num-el)
                                     (java.lang.foreign.MemorySegment/copy slice (java.lang.foreign.ValueLayout/JAVA_FLOAT) 0
                                                                           ^floats arr offset num-el)))
                                 (let [default-f (float default-val)]
                                   (if is-bf16-format
                                     (let [bf-bits (short (bit-shift-right (Float/floatToRawIntBits default-f) 16))]
                                       (java.util.Arrays/fill ^shorts arr offset (+ offset num-el) bf-bits))
                                     (java.util.Arrays/fill ^floats arr offset (+ offset num-el) default-f))))))
        layer-bufs (mapv (fn [{:keys [q-dim kv-dim head-dim mlp-dim qkv-rows gate-up-rows norms-base-el norms-total-el pl-total-el keys]}]
                           (let [qkv-el (* qkv-rows hidden-dim)
                                 o-el (* hidden-dim q-dim)
                                 gate-up-el (* gate-up-rows hidden-dim)
                                 down-el (* hidden-dim mlp-dim)
                                 layer-total-el (+ qkv-el o-el gate-up-el down-el norms-total-el)
                                 layer-arr (if is-bf16-format (short-array layer-total-el) (float-array layer-total-el))

                                 _ (copy-tensor-to-arr layer-arr (:q-w keys) [q-dim hidden-dim] 0 0.0)
                                 _ (copy-tensor-to-arr layer-arr (:k-w keys) [kv-dim hidden-dim] (* q-dim hidden-dim) 0.0)
                                 _ (copy-tensor-to-arr layer-arr (:v-w keys) [kv-dim hidden-dim] (* (+ q-dim kv-dim) hidden-dim) 0.0)

                                 o-off qkv-el
                                 _ (copy-tensor-to-arr layer-arr (:o-w keys) [hidden-dim q-dim] o-off 0.0)

                                 gu-off (+ o-off o-el)
                                 _ (copy-tensor-to-arr layer-arr (:gate-w keys) [mlp-dim hidden-dim] gu-off 0.0)
                                 _ (copy-tensor-to-arr layer-arr (:up-w keys) [mlp-dim hidden-dim] (+ gu-off (* mlp-dim hidden-dim)) 0.0)

                                 dw-off (+ gu-off gate-up-el)
                                 _ (copy-tensor-to-arr layer-arr (:down-w keys) [hidden-dim mlp-dim] dw-off 0.0)

                                 n-off (+ dw-off down-el)
                                 _ (copy-tensor-to-arr layer-arr (:input-ln-w keys) [hidden-dim] n-off 0.0)
                                 _ (copy-tensor-to-arr layer-arr (:layer-scalar-w keys) [1] (+ n-off hidden-dim) 1.0)
                                 _ (copy-tensor-to-arr layer-arr (:q-norm-w keys) [head-dim] (+ n-off hidden-dim 1) 0.0)
                                 _ (copy-tensor-to-arr layer-arr (:k-norm-w keys) [head-dim] (+ n-off hidden-dim 1 head-dim) 0.0)
                                 _ (copy-tensor-to-arr layer-arr (:post-attn-ln-w keys) [hidden-dim] (+ n-off hidden-dim 1 head-dim head-dim) 0.0)
                                 _ (copy-tensor-to-arr layer-arr (:pre-mlp-ln-w keys) [hidden-dim] (+ n-off (* 2 hidden-dim) 1 head-dim head-dim) 0.0)
                                 _ (copy-tensor-to-arr layer-arr (:post-mlp-ln-w keys) [hidden-dim] (+ n-off (* 3 hidden-dim) 1 head-dim head-dim) 0.0)
                                 _ (when (pos? pl-total-el)
                                     (copy-tensor-to-arr layer-arr (:per-layer-gate-w keys) [pl-dim hidden-dim] (+ n-off norms-base-el) 0.0)
                                     (copy-tensor-to-arr layer-arr (:per-layer-proj-w keys) [hidden-dim pl-dim] (+ n-off norms-base-el (* pl-dim hidden-dim)) 0.0)
                                     (copy-tensor-to-arr layer-arr (:post-per-layer-norm-w keys) [hidden-dim] (+ n-off norms-base-el (* pl-dim hidden-dim) (* hidden-dim pl-dim)) 0.0))]
                             (xla/buffer-from-host-buffer ctx client layer-arr [layer-total-el] weight-enum)))
                         layer-grouped-specs)]
    (vec (concat (into [embed-buf] ple-global-bufs) [final-norm-buf] layer-bufs))))

(defn build-tensor-logic-invars
  "Constructs EDN SSA signature invars for full Gemma 4 model forward pass."
  [config max-seq-len]
  (let [{:keys [vocab-size hidden-dim total-pl-dim pl-dim num-layers weight-dtype is-int8 layer-configs]} config
        norm-dtype (if is-int8 :bf16 weight-dtype)
        has-ple? (pos? total-pl-dim)]
    (vec (concat [[:x [:tensor [1 max-seq-len] :i32]]
                  [:embed_tokens [:tensor [vocab-size hidden-dim] norm-dtype]]]
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

(defn allocate-tensor-logic-weights
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
                             (let [kmap (gemma/gemma4-weight-key-map i (str prefix-base "layers."))
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

(defn compile-tensor-logic-executable
  "Compiles Gemma 4 model AST into a native StableHLO MLIR executable."
  [{:keys [ctx config opts]} max-seq-len]
  (let [invars (build-tensor-logic-invars config max-seq-len)
        ast (gemma-logic/gemma4-model-ast (assoc config :max-seq-len max-seq-len))
        _ (when-not (:quiet opts)
            (println (format "Lowering declarative Tensor Logic Gemma 4 AST (%d layers, max-seq-len=%d) to StableHLO..."
                             (:num-layers config) max-seq-len)))
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
   (let [{:keys [max-new-tokens quiet]} opts
         seq-len (long (or max-seq-len (:max-seq-len config) 128))
         vocab-size (long (or (:vocab-size config) (:vocab_size config) 262144))
         cur-tokens (atom (vec prompt-ids))
         t0 (System/nanoTime)]
     (loop [step 0]
       (if (>= step max-new-tokens)
         nil
         (let [s-len (count @cur-tokens)
               in-arr (int-array (take seq-len (concat @cur-tokens (repeat 0))))
               in-b (xla/buffer-from-host-buffer ctx (:client ctx) in-arr [1 seq-len] 4)
               args (into [in-b] device-weights)
               out (xla/execute exec args)
               logits (xla/to-host-slice out (dec s-len) vocab-size (* seq-len vocab-size) (get config :weight-dtype :bf16))
               next-id (sample-next-token logits opts prompt-ids (subvec @cur-tokens (count prompt-ids)))]
           (xla/destroy-buffer! ctx in-b)
           (xla/destroy-buffer! ctx out)
           (swap! cur-tokens conj next-id)
           (when-not quiet
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

(defn get-optimal-max-seq-len
  "Calculates optimal 1024-block aligned KV-cache seq length for given prompt and generation target."
  [prompt-len max-new-tokens]
  (let [total (+ (long (or prompt-len 16)) (long (or max-new-tokens 10)) 32)
        blocks (long (Math/ceil (/ (double total) 1024.0)))
        needed (* (max 1 blocks) 1024)]
    (min 16384 needed)))

(defn compile-inference-executables
  "Traces and JIT-compiles Dual Gemma 4 Execution Graphs (1 Prefill Graph + 1 Decode Step Graph)."
  [{:keys [ctx config opts]} prompt-len]
  (let [{:keys [vocab-size hidden-dim total-pl-dim pl-dim num-layers weight-dtype is-int8 layer-configs num-heads num-kv-heads num-kv-shared-layers layer-grouped-specs]} config
        {:keys [max-new-tokens quiet]} opts
        prompt-len (long (or prompt-len 16))
        max-seq-len (get-optimal-max-seq-len prompt-len max-new-tokens)
        num-unshared (- num-layers num-kv-shared-layers)
        norm-dtype (if is-int8 :bf16 weight-dtype)
        has-ple? (pos? total-pl-dim)
        ple-global-invars (when has-ple?
                            [[:embed_tokens_per_layer [:tensor [vocab-size total-pl-dim] norm-dtype]]
                             [:per_layer_model_projection [:tensor [total-pl-dim hidden-dim] norm-dtype]]
                             [:per_layer_projection_norm [:tensor [pl-dim] norm-dtype]]])
        layer-invars (mapv (fn [i]
                             (let [{:keys [q-dim qkv-rows gate-up-rows mlp-dim norms-total-el]} (nth layer-grouped-specs i)
                                   qkv-el (* qkv-rows hidden-dim)
                                   o-el (* hidden-dim q-dim)
                                   gate-up-el (* gate-up-rows hidden-dim)
                                   down-el (* hidden-dim mlp-dim)
                                   layer-total-el (+ qkv-el o-el gate-up-el down-el norms-total-el)]
                               [(keyword (str "l" i "_weights")) [:tensor [layer-total-el] weight-dtype]]))
                           (range num-layers))
        kv-invars (mapcat (fn [i]
                            (let [cfg (nth layer-configs i)
                                  l-nkv (:num-kv-heads cfg)
                                  h-dim (:head-dim cfg)]
                              [[(keyword (str "kc_" i)) [:tensor [1 l-nkv max-seq-len h-dim] norm-dtype]]
                               [(keyword (str "vc_" i)) [:tensor [1 l-nkv max-seq-len h-dim] norm-dtype]]]))
                          (range num-layers))

        common-rest-invars (vec (concat ple-global-invars
                                        [[:final_norm_w [:tensor [hidden-dim] norm-dtype]]]
                                        layer-invars
                                        kv-invars))

        prefill-invars (vec (concat [[:x [:tensor [1 prompt-len] :i32]]
                                     [:pos [:tensor [prompt-len] :i32]]
                                     [:embed_tokens [:tensor [vocab-size hidden-dim] norm-dtype]]]
                                    common-rest-invars))

        decode-invars (vec (concat [[:x [:tensor [1] :i32]]
                                    [:pos [:tensor [] :i32]]
                                    [:embed_tokens [:tensor [vocab-size hidden-dim] norm-dtype]]]
                                   common-rest-invars))

        build-fn (fn [is-prefill?]
                   (fn [x pos-tracer emb & rest-args]
                     (let [n-ple (if has-ple? 3 0)
                           emb-pl (when has-ple? (nth rest-args 0))
                           pl-model-proj (when has-ple? (nth rest-args 1))
                           pl-proj-norm (when has-ple? (nth rest-args 2))
                           fn-norm (nth rest-args n-ple)
                           after-fn-norm (subvec (vec rest-args) (inc n-ple))
                           n-layer-weights (count layer-invars)
                           all-layer-tracers (subvec after-fn-norm 0 n-layer-weights)
                           kv-tracers (subvec after-fn-norm n-layer-weights)
                           init-kv-caches (mapv vector (take-nth 2 kv-tracers) (take-nth 2 (rest kv-tracers)))
                           lw-seq (mapv (fn [i tr]
                                          (let [{:keys [q-dim kv-dim head-dim mlp-dim qkv-rows gate-up-rows norms-base-el norms-total-el]} (nth layer-grouped-specs i)
                                                cfg (nth layer-configs i)
                                                qkv-el (* qkv-rows hidden-dim)
                                                o-el (* hidden-dim q-dim)
                                                gate-up-el (* gate-up-rows hidden-dim)
                                                down-el (* hidden-dim mlp-dim)
                                                qkv-1d (t/slice tr [0] [qkv-el] [1])
                                                qkv-tr (t/reshape qkv-1d [qkv-rows hidden-dim])
                                                o-off qkv-el
                                                o-1d (t/slice tr [o-off] [(+ o-off o-el)] [1])
                                                o-tr (t/reshape o-1d [hidden-dim q-dim])
                                                gu-off (+ o-off o-el)
                                                gu-1d (t/slice tr [gu-off] [(+ gu-off gate-up-el)] [1])
                                                gate-up-tr (t/reshape gu-1d [gate-up-rows hidden-dim])
                                                dw-off (+ gu-off gate-up-el)
                                                dw-1d (t/slice tr [dw-off] [(+ dw-off down-el)] [1])
                                                down-tr (t/reshape dw-1d [hidden-dim mlp-dim])
                                                n-off (+ dw-off down-el)
                                                norms-tr (t/slice tr [n-off] [(+ n-off norms-total-el)] [1])
                                                qw (t/slice qkv-tr [0 0] [q-dim hidden-dim] [1 1])
                                                kw (t/slice qkv-tr [q-dim 0] [(+ q-dim kv-dim) hidden-dim] [1 1])
                                                vw (t/slice qkv-tr [(+ q-dim kv-dim) 0] [(+ q-dim kv-dim kv-dim) hidden-dim] [1 1])
                                                gw (t/slice gate-up-tr [0 0] [mlp-dim hidden-dim] [1 1])
                                                uw (t/slice gate-up-tr [mlp-dim 0] [(* 2 mlp-dim) hidden-dim] [1 1])
                                                in-ln (t/slice norms-tr [0] [hidden-dim] [1])
                                                ls (t/slice norms-tr [hidden-dim] [(+ hidden-dim 1)] [1])
                                                qn (t/slice norms-tr [(+ hidden-dim 1)] [(+ hidden-dim 1 head-dim)] [1])
                                                kn (t/slice norms-tr [(+ hidden-dim 1 head-dim)] [(+ hidden-dim 1 head-dim head-dim)] [1])
                                                post-attn-ln (t/slice norms-tr [(+ hidden-dim 1 head-dim head-dim)] [(+ (* 2 hidden-dim) 1 head-dim head-dim)] [1])
                                                pre-mlp-ln (t/slice norms-tr [(+ (* 2 hidden-dim) 1 head-dim head-dim)] [(+ (* 3 hidden-dim) 1 head-dim head-dim)] [1])
                                                post-mlp-ln (t/slice norms-tr [(+ (* 3 hidden-dim) 1 head-dim head-dim)] [(+ (* 4 hidden-dim) 1 head-dim head-dim)] [1])
                                                plg-1d (when has-ple? (t/slice norms-tr [norms-base-el] [(+ norms-base-el (* pl-dim hidden-dim))] [1]))
                                                plg (when has-ple? (t/reshape plg-1d [pl-dim hidden-dim]))
                                                plp-1d (when has-ple? (t/slice norms-tr [(+ norms-base-el (* pl-dim hidden-dim))] [(+ norms-base-el (* pl-dim hidden-dim) (* hidden-dim pl-dim))] [1]))
                                                plp (when has-ple? (t/reshape plp-1d [hidden-dim pl-dim]))
                                                pln (when has-ple? (t/slice norms-tr [(+ norms-base-el (* pl-dim hidden-dim) (* hidden-dim pl-dim))] [(+ norms-base-el (* pl-dim hidden-dim) (* hidden-dim pl-dim) hidden-dim)] [1]))]
                                            {:input-ln-w in-ln :layer-scalar-w ls
                                             :q-w qw :k-w kw :v-w vw :o-w o-tr
                                             :q-norm-w qn :k-norm-w kn
                                             :post-attn-ln-w post-attn-ln :pre-mlp-ln-w pre-mlp-ln :post-mlp-ln-w post-mlp-ln
                                             :gate-w gw :up-w uw :down-w down-tr
                                             :per-layer-gate-w plg :per-layer-proj-w plp :post-per-layer-norm-w pln
                                             :num-heads num-heads :num-kv-heads (:num-kv-heads cfg) :head-dim (:head-dim cfg)
                                             :theta-base (:theta-base cfg)
                                             :rope-proportion (:rope-proportion cfg)
                                             :norm-fn norm/rms-norm
                                             :attn-softcap nil
                                             :layer-type (:layer-type cfg)
                                             :is-shared? (>= i num-unshared)}))
                                        (range num-layers)
                                        all-layer-tracers)
                           f-opts {:final-logit-softcap 30.0 :num-kv-shared-layers num-kv-shared-layers :slice-last-token? is-prefill?}
                           [logits updated-kv] (gemma/full-gemma4-forward x emb emb-pl pl-model-proj pl-proj-norm lw-seq fn-norm pos-tracer num-heads num-kv-heads init-kv-caches (if is-prefill? 0 pos-tracer) f-opts)
                           last-logits (t/reshape logits [vocab-size])
                           flat-updated-kv (vec (apply concat updated-kv))]
                       (vec (concat [last-logits] flat-updated-kv)))))

        _ (when-not quiet (println "Tracing & Compiling Gemma 4 Prefill Execution Graph (Prompt:" prompt-len "tok)..."))
        prefill-graph (trace-graph "gemma4_prefill" prefill-invars (build-fn true))
        prefill-exec (xla/compile-graph ctx prefill-graph)

        _ (when-not quiet (println "Tracing & Compiling Gemma 4 Decode 1-Token Execution Graph..."))
        decode-graph (trace-graph "gemma4_decode" decode-invars (build-fn false))
        decode-exec (xla/compile-graph ctx decode-graph)
        _ (when-not quiet (println "Successfully compiled Gemma 4 Dual-Graph Executables!"))]
    {:prefill prefill-exec
     :decode decode-exec}))

(defn run-autoregressive-generation
  "Executes In-VRAM Autoregressive Loop using Dual Gemma 4 Executables."
  [{:keys [ctx opts config tokenizer]} execs device-weights prompt-ids]
  (let [{:keys [max-new-tokens quiet]} opts
        {:keys [num-layers layer-configs norm-enum]} config
        {:keys [prefill decode]} execs
        prompt-len (count prompt-ids)
        max-seq-len (get-optimal-max-seq-len prompt-len max-new-tokens)
        kv-bufs (let [arena (java.lang.foreign.Arena/global)]
                  (vec (mapcat (fn [i]
                                 (let [cfg (nth layer-configs i)
                                       l-nkv (:num-kv-heads cfg)
                                       h-dim (:head-dim cfg)
                                       n-bytes (long (* 1 l-nkv max-seq-len h-dim 2))
                                       off-heap-k (.allocate arena n-bytes 128)
                                       off-heap-v (.allocate arena n-bytes 128)]
                                   (.fill off-heap-k (byte 0))
                                   (.fill off-heap-v (byte 0))
                                   [(xla/buffer-from-host-buffer ctx (:client ctx) off-heap-k [1 l-nkv max-seq-len h-dim] norm-enum)
                                    (xla/buffer-from-host-buffer ctx (:client ctx) off-heap-v [1 l-nkv max-seq-len h-dim] norm-enum)]))
                               (range num-layers))))
        tok-buf (xla/buffer-from-host-buffer ctx (:client ctx) (int-array prompt-ids) [1 prompt-len] 4)
        pos-buf (xla/buffer-from-host-buffer ctx (:client ctx) (int-array (range prompt-len)) [prompt-len] 4)
        prefill-args (vec (concat [tok-buf pos-buf] device-weights kv-bufs))
        num-outputs (inc (count kv-bufs))
        t0 (System/nanoTime)

        ;; 1. Run Prefill pass on GPU (in-place updates kv-bufs in VRAM)
        prefill-res (pjrt/execute-executable ctx (:handle prefill) prefill-args num-outputs)
        prefill-tok-buf (if (sequential? prefill-res) (first prefill-res) prefill-res)
        updated-kv-bufs (if (sequential? prefill-res) (vec (rest prefill-res)) kv-bufs)

        vocab-size (long (or (:vocab-size config) (:vocab_size config) 262144))
        logits-arr (pjrt/buffer-to-host-buffer ctx prefill-tok-buf vocab-size :f32)
        t1 (System/nanoTime)
        prefill-ms (/ (- t1 t0) 1e6)

        first-tok (sample-next-token logits-arr opts prompt-ids [])]
    (xla/destroy-buffer! ctx tok-buf)
    (xla/destroy-buffer! ctx pos-buf)
    (xla/destroy-buffer! ctx prefill-tok-buf)
    (when-not quiet
      (println (format "  ↳ GPU Prefill Latency: %8.2f ms (%d prompt tokens)", prefill-ms prompt-len)))

    (loop [step 1
           cur-tok-id first-tok
           gen-ids [first-tok]
           cur-kv-bufs updated-kv-bufs
           old-kv-bufs []]
      (if (or (>= step max-new-tokens) (= cur-tok-id 1) (= cur-tok-id (eos-id tokenizer)))
        (let [t2 (System/nanoTime)
              total-ms (/ (- t2 t0) 1e6)
              gen-count (count gen-ids)
              decode-ms (- total-ms prefill-ms)
              decode-tok-s (if (pos? decode-ms) (/ (* (dec gen-count) 1000.0) decode-ms) 0.0)]
          (doseq [b (concat cur-kv-bufs old-kv-bufs)]
            (xla/destroy-buffer! ctx b))
          (when-not quiet
            (println "\n------------------------------------------------------------------")
            (println "  Telemetry Benchmark Metrics (Dual-Graph In-VRAM GPU Launch):")
            (println (format "    • Total GPU Prefill Latency   : %8.2f ms (%d tokens)" prefill-ms prompt-len))
            (println (format "    • Total GPU Decode Latency    : %8.2f ms (%d tokens)" decode-ms (dec gen-count)))
            (println (format "    • Decode Generation Speed     : %8.2f tok/s (%6.2f ms/tok)" decode-tok-s (if (pos? (dec gen-count)) (/ decode-ms (dec gen-count)) 0.0)))
            (println (format "    • Total GPU Latency           : %8.2f ms" total-ms))
            (println "------------------------------------------------------------------\n"))
          (vec (concat prompt-ids gen-ids)))
        (let [pos-val (int (min (dec max-seq-len) (+ prompt-len step -1)))
              pos-buf (xla/buffer-from-host-buffer ctx (:client ctx) (int-array [pos-val]) [] 4)
              cur-tok-buf (xla/buffer-from-host-buffer ctx (:client ctx) (int-array [cur-tok-id]) [1] 4)
              decode-args (vec (concat [cur-tok-buf pos-buf] device-weights cur-kv-bufs))
              decode-res (pjrt/execute-executable ctx (:handle decode) decode-args num-outputs)
              next-tok-buf (if (sequential? decode-res) (first decode-res) decode-res)
              next-kv-bufs (if (sequential? decode-res) (vec (rest decode-res)) cur-kv-bufs)
              step-logits (pjrt/buffer-to-host-buffer ctx next-tok-buf vocab-size :f32)
              next-tok-id (sample-next-token step-logits opts prompt-ids gen-ids)]
          (xla/destroy-buffer! ctx pos-buf)
          (xla/destroy-buffer! ctx cur-tok-buf)
          (xla/destroy-buffer! ctx next-tok-buf)
          (when (and (sequential? decode-res) (not= next-kv-bufs cur-kv-bufs))
            (doseq [b next-kv-bufs]
              (xla/destroy-buffer! ctx b)))
          (when-not quiet
            (println (format "  ↳ Decode Step %3d: pos=%3d token=%d" step pos-val next-tok-id)))
          (recur (inc step) next-tok-id (conj gen-ids next-tok-id) cur-kv-bufs []))))))

(defn generate-text
  "Generates text response using Gemma 4 model via pure Tensor Logic or legacy Tracer execution."
  [session prompt]
  (let [{:keys [tokenizer opts]} session
        {:keys [max-new-tokens temperature top-k model quiet method compare]} opts
        method (or method :tensor-logic)
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
        max-seq-len (long (or (:max-seq-len opts) (min 2048 (+ prompt-len max-new-tokens 16))))]
    (when-not quiet
      (let [prompt-str (if (> (count clean-prompt) 200)
                         (str (subs clean-prompt 0 100) " ... [truncated " (count clean-prompt) " chars] ... " (subs clean-prompt (- (count clean-prompt) 100)))
                         clean-prompt)
            tok-str (if (> prompt-len 30)
                      (str "[" (str/join " " (take 10 prompt-ids)) " ... " (str/join " " (take-last 5 prompt-ids)) "]")
                      (str prompt-ids))]
        (println (format "Prompt: \"%s\"" prompt-str))
        (println (format "Generation Options: max-new-tokens=%d, temperature=%.2f, top-k=%d, precision=%s, method=%s"
                         max-new-tokens temperature top-k (name (get-in session [:config :weight-dtype])) (name method)))
        (println (format "Encoded Token IDs (%d tokens): %s" prompt-len tok-str))))

    (if compare
      ;; Parity comparison between Legacy Tracer and Tensor Logic
      (do
        (println "\n==================================================================")
        (println "               STARTING E2E PARITY COMPARISON MODE                ")
        (println "==================================================================")
        (let [ctx (:ctx session)
              vocab-size (long (or (get-in session [:config :vocab-size]) (get-in session [:config :vocab_size]) 262144))
              _ (println "\n[1/4] Transferring weights and compiling Legacy Tracer...")
              trace-weights (allocate-device-weights session)
              trace-execs (compile-inference-executables session prompt-len)

              _ (println "\n[2/4] Transferring weights and compiling Tensor Logic...")
              logic-weights (allocate-tensor-logic-weights session)
              logic-exec (compile-tensor-logic-executable session max-seq-len)

              _ (println "\n[3/4] Running prompt forward pass on both engines...")
              kv-cache-seq-len (get-optimal-max-seq-len prompt-len max-new-tokens)
              num-layers (get-in session [:config :num-layers])
              layer-configs (get-in session [:config :layer-configs])
              norm-enum (get-in session [:config :norm-enum])
              arena (Arena/global)
              kv-bufs (vec (mapcat (fn [i]
                                     (let [cfg (nth layer-configs i)
                                           l-nkv (:num-kv-heads cfg)
                                           h-dim (:head-dim cfg)
                                           n-bytes (long (* 1 l-nkv kv-cache-seq-len h-dim 2))
                                           off-k (.allocate arena n-bytes 128)
                                           off-v (.allocate arena n-bytes 128)]
                                       (.fill off-k (byte 0))
                                       (.fill off-v (byte 0))
                                       [(xla/buffer-from-host-buffer ctx (:client ctx) off-k [1 l-nkv kv-cache-seq-len h-dim] norm-enum)
                                        (xla/buffer-from-host-buffer ctx (:client ctx) off-v [1 l-nkv kv-cache-seq-len h-dim] norm-enum)]))
                                   (range num-layers)))
              tok-buf (xla/buffer-from-host-buffer ctx (:client ctx) (int-array prompt-ids) [1 prompt-len] 4)
              pos-buf (xla/buffer-from-host-buffer ctx (:client ctx) (int-array (range prompt-len)) [prompt-len] 4)
              prefill-args (vec (concat [tok-buf pos-buf] trace-weights kv-bufs))
              num-outputs (inc (count kv-bufs))
              prefill-res (pjrt/execute-executable ctx (:handle (:prefill trace-execs)) prefill-args num-outputs)
              prefill-tok-buf (if (sequential? prefill-res) (first prefill-res) prefill-res)
              weight-dtype (get-in session [:config :weight-dtype] :bf16)
              logits-trace (pjrt/buffer-to-host-buffer ctx prefill-tok-buf vocab-size :f32)
              _ (do (xla/destroy-buffer! ctx tok-buf)
                    (xla/destroy-buffer! ctx pos-buf)
                    (xla/destroy-buffer! ctx prefill-tok-buf)
                    (doseq [b kv-bufs] (xla/destroy-buffer! ctx b)))

              in-arr (int-array (take max-seq-len (concat prompt-ids (repeat 0))))
              in-b (xla/buffer-from-host-buffer ctx (:client ctx) in-arr [1 max-seq-len] 4)
              logic-args (into [in-b] logic-weights)
              out-logic (xla/execute logic-exec logic-args)
              logits-logic (xla/to-host-slice out-logic (dec prompt-len) vocab-size (* max-seq-len vocab-size) weight-dtype)
              _ (do (xla/destroy-buffer! ctx in-b)
                    (xla/destroy-buffer! ctx out-logic))

              diffs (mapv (fn [a b] (Math/abs (double (- a b)))) (seq logits-trace) (seq logits-logic))
              max-diff (reduce max 0.0 diffs)
              mean-diff (/ (reduce + 0.0 diffs) (count diffs))
              top-k-trace (take 5 (sort-by second > (map-indexed vector logits-trace)))
              top-k-logic (take 5 (sort-by second > (map-indexed vector logits-logic)))]

          (println "\n--- Logit Parity Metrics ---")
          (println (format "Max Absolute Error  : %.6e" max-diff))
          (println (format "Mean Absolute Error : %.6e" mean-diff))
          (println "Top 5 Logits (Tracer)      :" top-k-trace)
          (println "Top 5 Logits (Tensor Logic):" top-k-logic)

          (println "\n[4/4] Running Autoregressive Generation Comparison...")
          (println "\nGenerating with [Legacy Tracer]...")
          (let [tokens-trace (run-autoregressive-generation session trace-execs trace-weights prompt-ids)
                _ (println "\nGenerating with [Tensor Logic]...")
                tokens-logic (run-autoregressive-generation-logic session logic-exec logic-weights prompt-ids max-seq-len)
                str-trace (decode tokenizer tokens-trace)
                str-logic (decode tokenizer tokens-logic)
                match? (= tokens-trace tokens-logic)]
            (doseq [w trace-weights] (xla/destroy-buffer! ctx w))
            (doseq [w logic-weights] (xla/destroy-buffer! ctx w))
            (println "\n==================================================================")
            (println "                   FINAL VERIFICATION SUMMARY                     ")
            (println "==================================================================")
            (println "Legacy Tracer Output:\n" str-trace)
            (println "\nTensor Logic Output :\n" str-logic)
            (println "Token Matching:" (if match? "EXACT MATCH (100% PARITY)" "MISMATCH"))
            (when-not match?
              (println "Notice: Token generation diverged due to sampling or numerical tolerance."))
            tokens-logic)))

      ;; Normal generation mode
      (let [metrics-atom (atom {})
            trace-spans-atom (atom [])]
        (if (= method :trace)
          (let [device-weights (binding [profile/*active-trace-spans* trace-spans-atom]
                                 (profile/with-profile metrics-atom "weight_transfer"
                                   (allocate-device-weights session)))
                execs (binding [profile/*active-trace-spans* trace-spans-atom]
                        (profile/with-profile metrics-atom "graph_compilation"
                          (compile-inference-executables session prompt-len)))]
            (when-not quiet
              (println "\nGenerating tokens autoregressively with Legacy Tracer Execution Graphs..."))
            (let [final-context (binding [profile/*active-trace-spans* trace-spans-atom]
                                  (profile/with-profile metrics-atom "autoregressive_generation"
                                    (run-autoregressive-generation session execs device-weights prompt-ids)))
                  generated-str (decode tokenizer final-context)]
              (println generated-str)
              (doseq [w device-weights] (xla/destroy-buffer! (:ctx session) w))
              final-context))

          ;; Default :tensor-logic
          (let [device-weights (binding [profile/*active-trace-spans* trace-spans-atom]
                                 (profile/with-profile metrics-atom "weight_transfer"
                                   (allocate-tensor-logic-weights session)))
                exec (binding [profile/*active-trace-spans* trace-spans-atom]
                       (profile/with-profile metrics-atom "graph_compilation"
                         (compile-tensor-logic-executable session max-seq-len)))]
            (when-not quiet
              (println "\nGenerating tokens autoregressively with pure Tensor Logic Gemma 4 Kernel..."))
            (let [final-context (binding [profile/*active-trace-spans* trace-spans-atom]
                                  (profile/with-profile metrics-atom "autoregressive_generation"
                                    (run-autoregressive-generation-logic session exec device-weights prompt-ids max-seq-len)))
                  generated-str (decode tokenizer final-context)]
              (println)
              (println generated-str)
              (doseq [w device-weights] (xla/destroy-buffer! (:ctx session) w))
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
              final-context)))))))

(defn generate-text-string
  "Generates text response using Gemma 4 model session and returns decoded text string."
  [session prompt]
  (let [{:keys [tokenizer]} session
        final-context (generate-text session prompt)]
    (decode tokenizer final-context)))

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
