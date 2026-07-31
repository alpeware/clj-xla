(ns scripts.benchmark
  "Latency and compilation throughput benchmarks."
  (:require [clj-xla.core :as xla]
            [clj-xla.models.gpt2 :as gpt2]
            [clj-xla.trace :refer [trace-graph]]))

(defn -main
  "Runs compilation benchmark."
  [& _args]
  (println "=== clj-xla Benchmark Suite ===")
  (let [ctx (xla/init-cpu!)
        graph (trace-graph "benchmark_block"
                           [[:x [:tensor [1 128 768] :f32]]
                            [:ln1_g [:tensor [768] :f32]]
                            [:ln1_b [:tensor [768] :f32]]
                            [:c_attn_w [:tensor [768 2304] :f32]]
                            [:c_attn_b [:tensor [2304] :f32]]
                            [:c_proj_w [:tensor [768 768] :f32]]
                            [:c_proj_b [:tensor [768] :f32]]
                            [:ln2_g [:tensor [768] :f32]]
                            [:ln2_b [:tensor [768] :f32]]
                            [:mlp_fc_w [:tensor [768 3072] :f32]]
                            [:mlp_fc_b [:tensor [3072] :f32]]
                            [:mlp_proj_w [:tensor [3072 768] :f32]]
                            [:mlp_proj_b [:tensor [768] :f32]]]
                           (fn [x ln1g ln1b cw cb pw pb ln2g ln2b fcw fcb pw2 pb2]
                             (gpt2/gpt2-block x {:ln1-g ln1g :ln1-b ln1b
                                                 :c-attn-w cw :c-attn-b cb
                                                 :c-proj-w pw :c-proj-b pb
                                                 :ln2-g ln2g :ln2-b ln2b
                                                 :mlp-fc-w fcw :mlp-fc-b fcb
                                                 :mlp-proj-w pw2 :mlp-proj-b pb2} 12)))
        t0 (System/nanoTime)
        exec1 (xla/compile-graph ctx graph)
        t1 (System/nanoTime)
        cold-ms (/ (- t1 t0) 1000000.0)
        t2 (System/nanoTime)
        exec2 (xla/compile-graph ctx graph)
        t3 (System/nanoTime)
        warm-us (/ (- t3 t2) 1000.0)]
    (println (format "Cold JIT Compilation: %.2f ms" cold-ms))
    (println (format "Warm Cached JIT Compilation: %.3f us" warm-us))
    (assert (= exec1 exec2) "Cached handle match"))
  (println "=== Benchmark Complete ==="))

(when (= *file* (System/getProperty "clojure.script.filename"))
  (apply -main *command-line-args*))
