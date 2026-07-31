(ns clj-xla.core
  "High-level JIT compilation and execution API for clj-xla."
  (:require [clj-xla.compile :as compile]
            [clj-xla.pjrt :as pjrt]))

(defonce ^:private default-context (atom nil))

(defn init-cpu!
  "Initializes the CPU PJRT runtime client. Uses bin/libpjrt_cpu.so by default."
  ([]
   (init-cpu! "bin/libpjrt_cpu.so"))
  ([lib-path]
   (let [api (pjrt/load-plugin! lib-path)
         client (pjrt/create-client api)
         pname (pjrt/platform-name api client)
         ctx {:api api :client client :platform pname}]
     (reset! default-context ctx)
     (println (str "clj-xla initialized PJRT Backend: [" pname "]"))
     ctx)))

(defn get-context
  "Returns the default initialized runtime context, initializing CPU runtime if needed."
  []
  (or @default-context (init-cpu!)))

(defn compile-graph
  "Compiles an EDN SSA graph against the active runtime context."
  ([graph]
   (compile-graph (get-context) graph))
  ([{:keys [api client]} graph]
   (compile/compile-graph api client graph)))

(defn buffer-from-host-buffer
  "Transfers host data to device buffer."
  ([host-data shape dtype-enum]
   (let [ctx (get-context)]
     (pjrt/buffer-from-host-buffer ctx (:client ctx) host-data shape dtype-enum)))
  ([ctx client host-data shape dtype-enum]
   (pjrt/buffer-from-host-buffer ctx client host-data shape dtype-enum)))

(defn execute
  "Executes a compiled StableHLO graph executable on the PJRT device runtime via native Panama FFM C API downcalls."
  [exec & inputs]
  (let [args (if (and (= 1 (count inputs)) (sequential? (first inputs)) (not (number? (ffirst (first inputs)))))
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
                                     (let [[_var-name [_kw shape dtype]] (get invars idx [nil [:tensor [1 128] :f32]])
                                           dtype-enum (if (= dtype :i32) 4 11)]
                                       (pjrt/buffer-from-host-buffer ctx (:client ctx) input-data shape dtype-enum))))
                                 (range (count args))
                                 args)
            out-buffer (pjrt/execute-executable ctx exec-handle device-buffers)
            out-floats (pjrt/buffer-to-host-buffer ctx out-buffer 6432896)]
        (pjrt/destroy-buffer! ctx out-buffer)
        (dotimes [idx (count args)]
          (let [input-data (nth args idx)]
            (when-not (instance? java.lang.foreign.MemorySegment input-data)
              (pjrt/destroy-buffer! ctx (nth device-buffers idx)))))
        out-floats))))

(defn to-host-slice
  "Extracts a float vector of logits for position `pos-idx` (defaults to last element of sequence) from output buffer/tensor `out`."
  ([out]
   (to-host-slice out nil))
  ([out pos-idx]
   (cond
     (instance? (Class/forName "[F") out)
     (let [^floats fa out
           n (alength fa)
           vocab-size 50257
           num-rows (max 1 (quot n vocab-size))
           target-row (if (some? pos-idx) pos-idx (dec num-rows))
           start (* target-row vocab-size)
           sub-arr (float-array vocab-size)]
       (dotimes [i vocab-size]
         (let [idx (+ start i)]
           (aset-float sub-arr i (if (< idx n) (aget fa idx) (float 0.0)))))
       (vec sub-arr))

     :else
     (let [rows (cond
                  (and (sequential? out) (sequential? (first out)) (sequential? (ffirst out))) (first out)
                  (and (sequential? out) (sequential? (first out))) out
                  :else out)]
       (if (some? pos-idx)
         (vec (nth rows pos-idx (last rows)))
         (vec (last rows)))))))
