(ns clj-xla.benchmark.runner
  "Execution engine for measuring JIT compilation throughput, execution latency, FLOPS, and memory bandwidth across PJRT backends."
  (:require [clj-xla.benchmark.core :as bcore]
            [clj-xla.benchmark.workloads :as bw]
            [clj-xla.core :as xla]
            [clj-xla.pjrt :as pjrt]))

(defn benchmark-workload
  "Measures cold compilation, warm compilation, latency percentiles, FLOPS, and bandwidth for `workload` on `ctx`."
  ([ctx workload] (benchmark-workload ctx workload {}))
  ([ctx workload opts]
   (binding [xla/*default-context* ctx]
     (let [{:keys [id name category flops bytes build-graph-fn make-inputs-fn]} workload
           warmup-iters (get opts :warmup-iters 5)
           measure-iters (get opts :measure-iters 25)
           target (or (:target ctx) :cpu)

           ;; 1. Measure Cold JIT compilation
           t0 (System/nanoTime)
           graph (build-graph-fn)
           exec1 (xla/compile-graph ctx graph)
           t1 (System/nanoTime)
           cold-ms (/ (- t1 t0) 1000000.0)

           ;; 2. Measure Warm cached compilation
           t2 (System/nanoTime)
           _exec2 (xla/compile-graph ctx graph)
           t3 (System/nanoTime)
           warm-us (/ (- t3 t2) 1000.0)

           ;; 3. Transfer input data to device buffers
           inputs (make-inputs-fn)
           invars (:invars graph)
           input-args (mapv (fn [idx [_var-name _]]
                              (get inputs _var-name (nth (vals inputs) idx)))
                            (range (count invars))
                            invars)

           ;; 4. Warmup passes
           _ (dotimes [_ warmup-iters]
               (let [out (xla/execute exec1 input-args)]
                 (if (vector? out)
                   (doseq [b out] (pjrt/destroy-buffer! ctx b))
                   (pjrt/destroy-buffer! ctx out))))

           ;; 5. Measured passes
           latencies (mapv (fn [_]
                             (let [start (System/nanoTime)
                                   out (xla/execute exec1 input-args)
                                   end (System/nanoTime)]
                               (if (vector? out)
                                 (doseq [b out] (pjrt/destroy-buffer! ctx b))
                                 (pjrt/destroy-buffer! ctx out))
                               (/ (- end start) 1000000.0)))
                           (range measure-iters))

           stats (bcore/calculate-latency-stats latencies)
           mean-lat (:mean stats)
           total-flops (if (fn? flops) (flops) (or flops 0))
           total-bytes (if (fn? bytes) (bytes) (or bytes 0))
           gflops (bcore/calculate-gflops total-flops mean-lat)
           tflops (bcore/calculate-tflops total-flops mean-lat)
           gbps (bcore/calculate-gbps total-bytes mean-lat)]
       (merge {:kernel id
               :name name
               :category category
               :backend target
               :cold-ms cold-ms
               :warm-us warm-us
               :mean-ms mean-lat
               :p50-ms (:p50 stats)
               :p90-ms (:p90 stats)
               :p99-ms (:p99 stats)
               :flops total-flops
               :bytes total-bytes
               :gflops gflops
               :tflops tflops
               :gbps gbps}
              stats)))))

(defn run-backend-benchmarks
  "Initializes `target-backend` and benchmarks all registered workloads."
  ([target-backend] (run-backend-benchmarks target-backend {}))
  ([target-backend opts]
   (let [ctx (xla/init-backend! target-backend)]
     (mapv (fn [[_id wl]]
             (benchmark-workload ctx wl opts))
           bw/WORKLOADS))))
