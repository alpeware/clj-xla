(ns clj-xla.core
  "High-level Public Clojure API for OpenXLA PJRT backend initialization, compilation, and execution."
  (:require [clj-xla.compile :as compile]
            [clj-xla.pjrt :as pjrt]))

(def ^:dynamic *default-context* nil)

(defn init-cpu!
  "Initializes PJRT CPU C API client runtime."
  []
  (let [lib-path (or (System/getenv "PJRT_CPU_LIBRARY_PATH") "bin/libpjrt_cpu.so")
        api-ctx (pjrt/load-plugin! lib-path)
        client (pjrt/create-client api-ctx)
        ctx (assoc api-ctx :client client)]
    (alter-var-root #'*default-context* (constantly ctx))
    ctx))

(defn get-context
  "Returns current default PJRT context or initializes CPU client."
  []
  (or *default-context* (init-cpu!)))

(defn compile-graph
  "Compiles EDN SSA graph to a native PJRT loaded executable."
  ([graph]
   (compile-graph (get-context) graph))
  ([ctx graph]
   (compile/compile-graph ctx (:client ctx) graph)))

(defn buffer-from-host-buffer
  "Transfers host primitive array/buffer into native PJRT device memory buffer."
  ([host-data shape dtype-enum]
   (let [ctx (get-context)]
     (pjrt/buffer-from-host-buffer ctx (:client ctx) host-data shape dtype-enum)))
  ([ctx client host-data shape dtype-enum]
   (pjrt/buffer-from-host-buffer ctx client host-data shape dtype-enum)))

(defn execute
  "Executes a compiled StableHLO graph executable on the PJRT device runtime via native Panama FFM C API downcalls."
  [exec & inputs]
  (let [flat-inputs (if (and (= 1 (count inputs)) (vector? (first inputs)))
                      (first inputs)
                      inputs)
        ctx (get-context)
        exec-handle (cond
                      (map? exec) (or (:handle exec) exec)
                      :else exec)]
    (if-not exec-handle
      (throw (ex-info "Invalid executable handle" {:exec exec}))
      (let [invars (or (get-in exec [:graph :invars]) [])
            device-buffers (mapv (fn [idx input-data]
                                   (if (instance? java.lang.foreign.MemorySegment input-data)
                                     input-data
                                     (let [[_var-name [_kw shape dtype]] (nth invars idx)
                                           dtype-enum (case dtype :i32 4 :f32 11 11)]
                                       (pjrt/buffer-from-host-buffer ctx (:client ctx) input-data shape dtype-enum))))
                                 (range (count flat-inputs))
                                 flat-inputs)
            out-buf (pjrt/execute-executable ctx exec-handle device-buffers)]
        out-buf))))

(defn to-host-slice
  "Transfers a slice of PJRT output device buffer back to host float array."
  ([out-buf]
   (to-host-slice out-buf 0 50257))
  ([out-buf slice-idx]
   (to-host-slice out-buf slice-idx 50257))
  ([out-buf slice-idx vocab-size]
   (let [ctx (get-context)
         total-floats (* 128 vocab-size)
         all-floats (pjrt/buffer-to-host-buffer ctx out-buf total-floats)
         offset (* slice-idx vocab-size)]
     (if (<= (+ offset vocab-size) (alength ^floats all-floats))
       (let [slice (float-array vocab-size)]
         (System/arraycopy all-floats offset slice 0 vocab-size)
         slice)
       all-floats))))
