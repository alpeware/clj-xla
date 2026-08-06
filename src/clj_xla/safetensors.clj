(ns clj-xla.safetensors
  "Panama FFM zero-copy off-heap .safetensors weight loader."
  (:require [clojure.data.json :as json]
            [clojure.java.io :as io])
  (:import [java.io RandomAccessFile]
           [java.lang.foreign Arena MemorySegment ValueLayout]
           [java.nio.channels FileChannel FileChannel$MapMode]
           [java.nio.file Path StandardOpenOption]))

(defn parse-header-json
  "Parses safetensors UTF-8 JSON header metadata string."
  [json-str]
  (json/read-str json-str))

(defn read-header
  "Reads the safetensors 8-byte unsigned header size and JSON metadata map from `file-path`."
  [file-path]
  (with-open [raf (RandomAccessFile. ^String file-path "r")]
    (let [header-size (.readLong raf)
          header-size-le (Long/reverseBytes header-size)
          header-bytes (byte-array header-size-le)]
      (.readFully raf header-bytes)
      (let [json-str (String. header-bytes "UTF-8")]
        {:header-size header-size-le
         :metadata (parse-header-json json-str)}))))

(declare map-safetensors-weights)

(defn map-safetensors-directory
  "Memory-maps all `.safetensors` files in `dir-path` using `arena`."
  [dir-path arena]
  (let [dir (io/file dir-path)
        safetensor-files (filter #(and (.isFile %) (.endsWith (.getName %) ".safetensors"))
                                 (.listFiles dir))
        mapped-shards (mapv (fn [f]
                              (let [path (.getAbsolutePath f)
                                    {:keys [header segment]} (map-safetensors-weights path arena)]
                                {:header header :segment segment :file-path path}))
                            safetensor-files)
        tensor-map (into {}
                         (for [shard mapped-shards
                               [t-name t-info] (:header shard)]
                           [t-name {:info t-info :segment (:segment shard)}]))
        header-map (into {} (map (fn [[k v]] [k (:info v)]) tensor-map))]
    {:header header-map
     :tensors tensor-map
     :shards mapped-shards}))

(defn map-safetensors-weights
  "Maps `file-path` (or directory) off-heap into Java 25 Panama MemorySegments using `arena`.
   Returns {:header header-metadata :segment memory-segment} or {:header ... :tensors ...}."
  [^String file-path ^Arena arena]
  (let [f (io/file file-path)]
    (cond
      (.isDirectory f)
      (map-safetensors-directory file-path arena)

      (and (not (.exists f))
           (some? (.getParent f))
           (.exists (io/file (.getParent f) "model-00001-of-00002.safetensors")))
      (map-safetensors-directory (.getParent f) arena)

      :else
      (let [{:keys [header-size metadata]} (read-header file-path)
            path (Path/of file-path (into-array String []))
            fc (FileChannel/open path (into-array [StandardOpenOption/READ]))
            total-len (.size fc)
            weight-bytes-len (- total-len (+ 8 header-size))
            ^MemorySegment mapped-seg (.map fc FileChannel$MapMode/READ_ONLY (+ 8 (long header-size)) (long weight-bytes-len) arena)]
        {:header metadata
         :segment mapped-seg}))))

(defn get-tensor-slice
  "Extracts off-heap memory segment slice for tensor `tensor-name` from mapped weights."
  [mapped-weights tensor-name]
  (if-let [t-entry (get (:tensors mapped-weights) tensor-name)]
    (let [{:keys [info segment]} t-entry
          [start end] (get info "data_offsets")
          len (- end start)]
      (.asSlice ^MemorySegment segment (long start) (long len)))
    (if-let [tensor-info (get (:header mapped-weights) tensor-name)]
      (let [segment (:segment mapped-weights)
            [start end] (get tensor-info "data_offsets")
            len (- end start)]
        (.asSlice ^MemorySegment segment (long start) (long len)))
      (throw (ex-info "Tensor key not found in Safetensors header" {:tensor tensor-name})))))

(defn get-tensor-floats
  "Reads Float32 or BF16 tensor values from `mapped-weights` into a Java float array using Little-Endian byte order."
  [mapped-weights tensor-name]
  (let [t-entry (or (get (:tensors mapped-weights) tensor-name)
                    (when-let [info (get (:header mapped-weights) tensor-name)]
                      {:info info :segment (:segment mapped-weights)}))]
    (if-not t-entry
      (throw (ex-info "Tensor key not found in Safetensors header" {:tensor tensor-name}))
      (let [{:keys [info segment]} t-entry
            [start end] (get info "data_offsets")
            dtype (get info "dtype" "F32")
            len (- end start)
            ^MemorySegment slice (.asSlice ^MemorySegment segment (long start) (long len))]
        (cond
          (or (= dtype "BF16") (= dtype "BFLOAT16"))
          (let [num-elements (quot len 2)
                chunk-size 50000000
                target-len (min num-elements Integer/MAX_VALUE)
                arr (float-array target-len)]
            (loop [offset 0]
              (when (< offset target-len)
                (let [cur-len (min chunk-size (- target-len offset))
                      sa (short-array cur-len)
                      byte-offset (* (long offset) 2)
                      seg-slice (.asSlice slice byte-offset (* (long cur-len) 2))]
                  (MemorySegment/copy seg-slice ValueLayout/JAVA_SHORT (long 0) sa (long 0) (long cur-len))
                  (dotimes [i cur-len]
                    (let [s (int (aget sa i))
                          bits (unchecked-int (bit-shift-left (long (bit-and s 0xffff)) 16))]
                      (aset arr (+ offset i) (Float/intBitsToFloat bits))))
                  (recur (+ offset cur-len)))))
            arr)

          :else
          (let [num-floats (quot len 4)
                arr (float-array (min num-floats Integer/MAX_VALUE))]
            (MemorySegment/copy slice ValueLayout/JAVA_FLOAT (long 0) arr (long 0) (long (count arr)))
            arr))))))

(defn get-tensor-bf16-shorts
  "Reads BF16 or Float32 tensor values from `mapped-weights` into a Java short array (BF16 raw 16-bit values)."
  [mapped-weights tensor-name]
  (let [t-entry (or (get (:tensors mapped-weights) tensor-name)
                    (when-let [info (get (:header mapped-weights) tensor-name)]
                      {:info info :segment (:segment mapped-weights)}))]
    (if-not t-entry
      (throw (ex-info "Tensor key not found in Safetensors header" {:tensor tensor-name}))
      (let [{:keys [info segment]} t-entry
            [start end] (get info "data_offsets")
            dtype (get info "dtype" "F32")
            len (- end start)
            ^MemorySegment slice (.asSlice ^MemorySegment segment (long start) (long len))]
        (cond
          (or (= dtype "BF16") (= dtype "BFLOAT16"))
          (let [num-shorts (quot len 2)
                arr (short-array num-shorts)]
            (MemorySegment/copy slice ValueLayout/JAVA_SHORT (long 0) arr (long 0) (long num-shorts))
            arr)

          :else
          (let [num-floats (quot len 4)
                fa (float-array num-floats)
                _ (MemorySegment/copy slice ValueLayout/JAVA_FLOAT (long 0) fa (long 0) (long num-floats))
                arr (short-array num-floats)]
            (dotimes [i num-floats]
              (let [f (aget fa i)
                    bits (Float/floatToIntBits f)
                    s (short (bit-shift-right bits 16))]
                (aset arr i s)))
            arr))))))
