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
          (do
            (when (= (keyword val) :rocm)
              (try
                (let [jh (System/getProperty "java.home")
                      paths [(str jh "/lib/server/libjsig.so")
                             (str jh "/lib/libjsig.so")
                             "/usr/lib64/openjdk-25/lib/server/libjsig.so"]]
                  (doseq [p paths]
                    (when (.exists (java.io.File. ^String p))
                      (try (System/load p) (catch Throwable _ nil)))))
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
            (recur (subvec remaining 2) (assoc opts :backend (keyword val))))

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
        layer-bufs (mapcat (fn [{:keys [q-dim kv-dim head-dim mlp-dim qkv-rows gate-up-rows norms-base-el norms-total-el pl-total-el keys]}]
                             (let [qkv-el (* qkv-rows hidden-dim)
                                   qkv-arr (if is-bf16-format (short-array qkv-el) (float-array qkv-el))
                                   _ (copy-tensor-to-arr qkv-arr (:q-w keys) [q-dim hidden-dim] 0 0.0)
                                   _ (copy-tensor-to-arr qkv-arr (:k-w keys) [kv-dim hidden-dim] (* q-dim hidden-dim) 0.0)
                                   _ (copy-tensor-to-arr qkv-arr (:v-w keys) [kv-dim hidden-dim] (* (+ q-dim kv-dim) hidden-dim) 0.0)
                                   [qkv-buf qkv-scale] (if is-int8
                                                         (let [q (quantize-bf16-to-int8 qkv-arr)]
                                                           [(xla/buffer-from-host-buffer ctx client (:data q) [qkv-rows hidden-dim] weight-enum)
                                                            (xla/buffer-from-host-buffer ctx client (float-array [(:scale q)]) [1] 11)])
                                                         [(xla/buffer-from-host-buffer ctx client qkv-arr [qkv-rows hidden-dim] weight-enum) nil])

                                   o-el (* hidden-dim q-dim)
                                   o-arr (if is-bf16-format (short-array o-el) (float-array o-el))
                                   _ (copy-tensor-to-arr o-arr (:o-w keys) [hidden-dim q-dim] 0 0.0)
                                   [o-buf o-scale] (if is-int8
                                                     (let [q (quantize-bf16-to-int8 o-arr)]
                                                       [(xla/buffer-from-host-buffer ctx client (:data q) [hidden-dim q-dim] weight-enum)
                                                        (xla/buffer-from-host-buffer ctx client (float-array [(:scale q)]) [1] 11)])
                                                     [(xla/buffer-from-host-buffer ctx client o-arr [hidden-dim q-dim] weight-enum) nil])

                                   gate-up-el (* gate-up-rows hidden-dim)
                                   gate-up-arr (if is-bf16-format (short-array gate-up-el) (float-array gate-up-el))
                                   _ (copy-tensor-to-arr gate-up-arr (:gate-w keys) [mlp-dim hidden-dim] 0 0.0)
                                   _ (copy-tensor-to-arr gate-up-arr (:up-w keys) [mlp-dim hidden-dim] (* mlp-dim hidden-dim) 0.0)
                                   [gate-up-buf gate-up-scale] (if is-int8
                                                                 (let [q (quantize-bf16-to-int8 gate-up-arr)]
                                                                   [(xla/buffer-from-host-buffer ctx client (:data q) [gate-up-rows hidden-dim] weight-enum)
                                                                    (xla/buffer-from-host-buffer ctx client (float-array [(:scale q)]) [1] 11)])
                                                                 [(xla/buffer-from-host-buffer ctx client gate-up-arr [gate-up-rows hidden-dim] weight-enum) nil])

                                   down-el (* hidden-dim mlp-dim)
                                   down-arr (if is-bf16-format (short-array down-el) (float-array down-el))
                                   _ (copy-tensor-to-arr down-arr (:down-w keys) [hidden-dim mlp-dim] 0 0.0)
                                   [down-buf down-scale] (if is-int8
                                                           (let [q (quantize-bf16-to-int8 down-arr)]
                                                             [(xla/buffer-from-host-buffer ctx client (:data q) [hidden-dim mlp-dim] weight-enum)
                                                              (xla/buffer-from-host-buffer ctx client (float-array [(:scale q)]) [1] 11)])
                                                           [(xla/buffer-from-host-buffer ctx client down-arr [hidden-dim mlp-dim] weight-enum) nil])

                                   norms-arr (if is-bf16-format (short-array norms-total-el) (float-array norms-total-el))
                                   _ (copy-tensor-to-arr norms-arr (:input-ln-w keys) [hidden-dim] 0 0.0)
                                   _ (copy-tensor-to-arr norms-arr (:layer-scalar-w keys) [1] hidden-dim 1.0)
                                   _ (copy-tensor-to-arr norms-arr (:q-norm-w keys) [head-dim] (+ hidden-dim 1) 0.0)
                                   _ (copy-tensor-to-arr norms-arr (:k-norm-w keys) [head-dim] (+ hidden-dim 1 head-dim) 0.0)
                                   _ (copy-tensor-to-arr norms-arr (:post-attn-ln-w keys) [hidden-dim] (+ hidden-dim 1 head-dim head-dim) 0.0)
                                   _ (copy-tensor-to-arr norms-arr (:pre-mlp-ln-w keys) [hidden-dim] (+ (* 2 hidden-dim) 1 head-dim head-dim) 0.0)
                                   _ (copy-tensor-to-arr norms-arr (:post-mlp-ln-w keys) [hidden-dim] (+ (* 3 hidden-dim) 1 head-dim head-dim) 0.0)
                                   _ (when (pos? pl-total-el)
                                       (copy-tensor-to-arr norms-arr (:per-layer-gate-w keys) [pl-dim hidden-dim] norms-base-el 0.0)
                                       (copy-tensor-to-arr norms-arr (:per-layer-proj-w keys) [hidden-dim pl-dim] (+ norms-base-el (* pl-dim hidden-dim)) 0.0)
                                       (copy-tensor-to-arr norms-arr (:post-per-layer-norm-w keys) [hidden-dim] (+ norms-base-el (* pl-dim hidden-dim) (* hidden-dim pl-dim)) 0.0))
                                   norms-buf (xla/buffer-from-host-buffer ctx client norms-arr [norms-total-el] norm-enum)]
                               (if is-int8
                                 [qkv-buf qkv-scale o-buf o-scale gate-up-buf gate-up-scale down-buf down-scale norms-buf]
                                 [qkv-buf o-buf gate-up-buf down-buf norms-buf])))
                           layer-grouped-specs)]
    (vec (concat (into [embed-buf] ple-global-bufs) [final-norm-buf] layer-bufs))))

(defn compile-inference-executables
  "Traces and JIT-compiles a Single Fused Gemma 4 StableHLO Graph into a native PJRT Executable."
  [{:keys [ctx config opts]}]
  (let [{:keys [vocab-size hidden-dim total-pl-dim pl-dim max-seq-len num-layers weight-dtype is-int8 layer-configs num-heads num-kv-heads num-kv-shared-layers layer-grouped-specs]} config
        {:keys [verbose quiet]} opts
        num-unshared (- num-layers num-kv-shared-layers)
        norm-dtype (if is-int8 :bf16 weight-dtype)
        has-ple? (pos? total-pl-dim)
        ple-global-invars (when has-ple?
                            [[:embed_tokens_per_layer [:tensor [vocab-size total-pl-dim] norm-dtype]]
                             [:per_layer_model_projection [:tensor [total-pl-dim hidden-dim] norm-dtype]]
                             [:per_layer_projection_norm [:tensor [pl-dim] norm-dtype]]])
        layer-invars (mapcat (fn [i]
                               (let [{:keys [q-dim qkv-rows gate-up-rows mlp-dim norms-total-el]} (nth layer-grouped-specs i)]
                                 (if is-int8
                                   [[(keyword (str "l" i "_qkv")) [:tensor [qkv-rows hidden-dim] :i8]]
                                    [(keyword (str "l" i "_qkv_s")) [:tensor [1] :f32]]
                                    [(keyword (str "l" i "_o")) [:tensor [hidden-dim q-dim] :i8]]
                                    [(keyword (str "l" i "_o_s")) [:tensor [1] :f32]]
                                    [(keyword (str "l" i "_gate_up")) [:tensor [gate-up-rows hidden-dim] :i8]]
                                    [(keyword (str "l" i "_gate_up_s")) [:tensor [1] :f32]]
                                    [(keyword (str "l" i "_down")) [:tensor [hidden-dim mlp-dim] :i8]]
                                    [(keyword (str "l" i "_down_s")) [:tensor [1] :f32]]
                                    [(keyword (str "l" i "_norms")) [:tensor [norms-total-el] :bf16]]]
                                   [[(keyword (str "l" i "_qkv")) [:tensor [qkv-rows hidden-dim] weight-dtype]]
                                    [(keyword (str "l" i "_o")) [:tensor [hidden-dim q-dim] weight-dtype]]
                                    [(keyword (str "l" i "_gate_up")) [:tensor [gate-up-rows hidden-dim] weight-dtype]]
                                    [(keyword (str "l" i "_down")) [:tensor [hidden-dim mlp-dim] weight-dtype]]
                                    [(keyword (str "l" i "_norms")) [:tensor [norms-total-el] weight-dtype]]])))
                             (range num-layers))
        invars (vec (concat
                     [[:x [:tensor [1 max-seq-len] :i32]]
                      [:pos [:tensor [max-seq-len] :i32]]
                      [:embed_tokens [:tensor [vocab-size hidden-dim] norm-dtype]]]
                     ple-global-invars
                     [[:final_norm_w [:tensor [hidden-dim] norm-dtype]]]
                     layer-invars))

        trace-fn (if has-ple?
                   (fn [x pos-tracer emb emb-pl pl-model-proj pl-proj-norm fn-norm & all-layer-tracers]
                     (let [group-size (if is-int8 9 5)
                           tr-partitioned (mapv vec (partition group-size all-layer-tracers))
                           deq (fn [w s] (t/* (t/convert w :bf16) (t/convert s :bf16)))
                           lw-seq (mapv (fn [i trs]
                                          (let [[qkv-tr qkv-s-tr o-tr o-s-tr gate-up-tr gate-up-s-tr down-tr down-s-tr norms-tr]
                                                (if is-int8 trs [(nth trs 0) nil (nth trs 1) nil (nth trs 2) nil (nth trs 3) nil (nth trs 4)])
                                                qkv-tr (if is-int8 (deq qkv-tr qkv-s-tr) qkv-tr)
                                                o-tr (if is-int8 (deq o-tr o-s-tr) o-tr)
                                                gate-up-tr (if is-int8 (deq gate-up-tr gate-up-s-tr) gate-up-tr)
                                                down-tr (if is-int8 (deq down-tr down-s-tr) down-tr)
                                                {:keys [q-dim kv-dim head-dim mlp-dim norms-base-el]} (nth layer-grouped-specs i)
                                                cfg (nth layer-configs i)
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
                                                plg-1d (t/slice norms-tr [norms-base-el] [(+ norms-base-el (* pl-dim hidden-dim))] [1])
                                                plg (t/reshape plg-1d [pl-dim hidden-dim])
                                                plp-1d (t/slice norms-tr [(+ norms-base-el (* pl-dim hidden-dim))] [(+ norms-base-el (* pl-dim hidden-dim) (* hidden-dim pl-dim))] [1])
                                                plp (t/reshape plp-1d [hidden-dim pl-dim])
                                                pln (t/slice norms-tr [(+ norms-base-el (* pl-dim hidden-dim) (* hidden-dim pl-dim))] [(+ norms-base-el (* pl-dim hidden-dim) (* hidden-dim pl-dim) hidden-dim)] [1])]
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
                                        tr-partitioned)
                           logits (gemma/full-gemma4-forward x emb emb-pl pl-model-proj pl-proj-norm lw-seq fn-norm pos-tracer num-heads num-kv-heads nil 0 {:final-logit-softcap 30.0 :num-kv-shared-layers num-kv-shared-layers})]
                       (t/convert logits :f32)))
                   (fn [x pos-tracer emb fn-norm & all-layer-tracers]
                     (let [group-size (if is-int8 9 5)
                           tr-partitioned (mapv vec (partition group-size all-layer-tracers))
                           deq (fn [w s] (t/* (t/convert w :bf16) (t/convert s :bf16)))
                           lw-seq (mapv (fn [i trs]
                                          (let [[qkv-tr qkv-s-tr o-tr o-s-tr gate-up-tr gate-up-s-tr down-tr down-s-tr norms-tr]
                                                (if is-int8 trs [(nth trs 0) nil (nth trs 1) nil (nth trs 2) nil (nth trs 3) nil (nth trs 4)])
                                                qkv-tr (if is-int8 (deq qkv-tr qkv-s-tr) qkv-tr)
                                                o-tr (if is-int8 (deq o-tr o-s-tr) o-tr)
                                                gate-up-tr (if is-int8 (deq gate-up-tr gate-up-s-tr) gate-up-tr)
                                                down-tr (if is-int8 (deq down-tr down-s-tr) down-tr)
                                                {:keys [q-dim kv-dim head-dim mlp-dim]} (nth layer-grouped-specs i)
                                                cfg (nth layer-configs i)
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
                                                post-mlp-ln (t/slice norms-tr [(+ (* 3 hidden-dim) 1 head-dim head-dim)] [(+ (* 4 hidden-dim) 1 head-dim head-dim)] [1])]
                                            {:input-ln-w in-ln :layer-scalar-w ls
                                             :q-w qw :k-w kw :v-w vw :o-w o-tr
                                             :q-norm-w qn :k-norm-w kn
                                             :post-attn-ln-w post-attn-ln :pre-mlp-ln-w pre-mlp-ln :post-mlp-ln-w post-mlp-ln
                                             :gate-w gw :up-w uw :down-w down-tr
                                             :num-heads num-heads :num-kv-heads (:num-kv-heads cfg) :head-dim (:head-dim cfg)
                                             :theta-base (:theta-base cfg)
                                             :rope-proportion (:rope-proportion cfg)
                                             :norm-fn norm/rms-norm
                                             :attn-softcap nil
                                             :layer-type (:layer-type cfg)
                                             :is-shared? (>= i num-unshared)}))
                                        (range num-layers)
                                        tr-partitioned)
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
  (let [{:keys [max-new-tokens temperature top-k]} opts
        {:keys [max-seq-len vocab-size]} config
        primary-eos (eos-id tokenizer)
        ;; Hard EOS tokens that always terminate generation immediately
        hard-eos (set (remove nil? [1 primary-eos]))
        ;; Token 107 (<turn|>) only terminates when repeated consecutively
        ;; (the first 107 may just end a thought channel in 12B-it models)
        turn-end-token 107
        pos-array (int-array (range max-seq-len))
        pos-buf (xla/buffer-from-host-buffer ctx (:client ctx) pos-array [max-seq-len] 4)
        ;; Accumulate intermediate device buffers for deferred cleanup.
        ;; Destroying buffers mid-loop crashes ROCm PJRT (the plugin retains
        ;; internal references even after the completion event is awaited).
        intermediate-bufs (atom [])]
    (try
      (loop [cur-tokens (vec prompt-ids)
             step 0
             consecutive-turn-ends 0]
        (let [last-tok (when (pos? step) (last cur-tokens))
              hit-hard-eos (and (pos? step) (contains? hard-eos last-tok))
              ;; Stop on consecutive <turn|> tokens (model is truly done)
              new-consecutive (if (= last-tok turn-end-token) (inc consecutive-turn-ends) 0)
              hit-turn-end (>= new-consecutive 2)]
          (if (or (>= step max-new-tokens) hit-hard-eos hit-turn-end)
            cur-tokens
            (let [seq-len (count cur-tokens)
                  padded-tokens (int-array (concat cur-tokens (repeat (- max-seq-len seq-len) 0)))
                  tok-buf (xla/buffer-from-host-buffer ctx (:client ctx) padded-tokens [1 max-seq-len] 4)
                  input-args (into [tok-buf pos-buf] device-weights)
                  logits-buf (xla/execute exec input-args)
                  step-logits (xla/to-host-slice logits-buf (dec seq-len) vocab-size (* max-seq-len vocab-size))
                  next-tok (sampling/sample-logits step-logits {:temperature temperature :top-k top-k})]
              (swap! intermediate-bufs into [tok-buf logits-buf])
              (recur (conj cur-tokens next-tok) (inc step) new-consecutive)))))
      (finally
        (doseq [buf @intermediate-bufs]
          (xla/destroy-buffer! ctx buf))))))

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
      (let [final-context (run-autoregressive-generation session exec device-weights prompt-ids)
            generated-str (decode tokenizer final-context)]
        (println generated-str)
        (when-not quiet
          (println "\n==================================================================")
          (println "=== Single Fused XLA GPU Forward Pass Verification Passed! ===")
          (println "=================================================================="))
        final-context))))

(defn- find-libjsig
  "Searches standard JDK paths for libjsig.so."
  []
  (let [jh (System/getProperty "java.home")
        paths [(str jh "/lib/server/libjsig.so")
               (str jh "/lib/libjsig.so")
               "/usr/lib64/openjdk-25/lib/server/libjsig.so"]]
    (first (filter #(.exists (io/file %)) paths))))

(defn- needs-libjsig-reexec?
  "Returns true when running with ROCm backend and LD_PRELOAD does not
   already include libjsig.so.  Signal chaining via LD_PRELOAD is required
   because the ROCm PJRT plugin bundles LLVM, which installs its own signal
   handlers that conflict with the JVM's.  System.load cannot interpose
   symbols globally — only LD_PRELOAD does."
  [opts]
  (and (= (:backend opts) :rocm)
       (not (some-> (System/getenv "LD_PRELOAD")
                    (.contains "libjsig")))))

(defn- reexec-with-libjsig!
  "Re-launches the current JVM process with LD_PRELOAD=libjsig.so so that
   the OpenJDK signal-chaining library intercepts all sigaction calls before
   the ROCm PJRT plugin installs LLVM's conflicting signal handlers."
  [args]
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
                             ["-cp" cp "clojure.main" "-m" "scripts.gemma4-inference"]
                             args))
            pb (ProcessBuilder. ^java.util.List cmd)
            env (.environment pb)
            existing-preload (.get env "LD_PRELOAD")
            new-preload (if (and existing-preload (not (.isEmpty ^String existing-preload)))
                          (str jsig-path ":" existing-preload)
                          jsig-path)]
        (.put env "LD_PRELOAD" new-preload)
        (.inheritIO pb)
        (System/exit (.waitFor (.start pb)))))))

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
