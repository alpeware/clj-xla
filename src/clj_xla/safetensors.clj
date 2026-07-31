(ns clj-xla.safetensors
  "Panama FFM zero-copy off-heap .safetensors weight loader."
  (:require [clojure.data.json :as json])
  (:import [java.io RandomAccessFile]
           [java.lang.foreign Arena MemorySegment]
           [java.nio FloatBuffer]
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

(defn map-safetensors-weights
  "Maps `file-path` off-heap into a Java 25 Panama MemorySegment using `arena`.
   Returns {:header header-metadata :segment memory-segment}."
  [^String file-path ^Arena arena]
  (let [{:keys [header-size metadata]} (read-header file-path)
        path (Path/of file-path (into-array String []))
        fc (FileChannel/open path (into-array [StandardOpenOption/READ]))
        total-len (.size fc)
        weight-bytes-len (- total-len (+ 8 header-size))
        ^MemorySegment mapped-seg (.map fc FileChannel$MapMode/READ_ONLY (+ 8 (long header-size)) (long weight-bytes-len) arena)]
    {:header metadata
     :segment mapped-seg}))

(defn get-tensor-slice
  "Extracts off-heap memory segment slice for tensor `tensor-name` from mapped weights."
  [{:keys [header segment]} tensor-name]
  (if-let [tensor-info (get header tensor-name)]
    (let [[start end] (get tensor-info "data_offsets")
          len (- end start)]
      (.asSlice ^MemorySegment segment (long start) (long len)))
    (throw (ex-info "Tensor key not found in Safetensors header" {:tensor tensor-name}))))

(defn get-tensor-floats
  "Reads Float32 tensor values from `mapped-weights` into a Java float array."
  [{:keys [header segment]} tensor-name]
  (if-let [tensor-info (get header tensor-name)]
    (let [[start end] (get tensor-info "data_offsets")
          len (- end start)
          num-floats (quot len 4)
          ^MemorySegment slice (.asSlice ^MemorySegment segment (long start) (long len))
          ^FloatBuffer float-buf (.asFloatBuffer (.asByteBuffer slice))
          arr (float-array num-floats)]
      (.get float-buf arr)
      arr)
    (throw (ex-info "Tensor key not found in Safetensors header" {:tensor tensor-name}))))
