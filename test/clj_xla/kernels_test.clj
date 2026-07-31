(ns clj-xla.kernels-test
  "Unit and generative tests for high-level neural network kernels."
  (:refer-clojure :exclude [+ * - /])
  (:require [clj-xla.compile :as compile]
            [clj-xla.example.kernels :as k]
            [clj-xla.pjrt :as pjrt]
            [clj-xla.trace :refer [trace-graph]]
            [clojure.test :refer [deftest is testing]]
            [clojure.test.check.clojure-test :refer [defspec]]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]))

(defspec prop-gelu-kernel-tracing 50
  (prop/for-all [_seq-len (gen/choose 1 64)]
                (let [graph (trace-graph "gelu_test" [[:x [:tensor [1 128 768] :f32]]] k/gelu)]
                  (and (= "gelu_test" (:name graph))
                       (seq (:eqns graph))))))

(deftest layer-norm-kernel-trace-test
  (testing "Symbolic tracing of pure Clojure layer-norm kernel"
    (let [ln-fn (fn [x g b] (k/layer-norm x g b 1e-5))
          graph (trace-graph "layer_norm"
                             [[:x [:tensor [1 128 768] :f32]]
                              [:gamma [:tensor [768] :f32]]
                              [:beta [:tensor [768] :f32]]]
                             ln-fn)]
      (is (= "layer_norm" (:name graph)))
      (is (pos? (count (:eqns graph)))))))

(deftest gelu-kernel-execution-test
  (testing "Tracing and compiling GELU kernel to hardware executable"
    (let [api (pjrt/load-plugin! "bin/libpjrt_cpu.so")
          client (pjrt/create-client api)
          graph (trace-graph "gelu_compilation" [[:x [:tensor [1 128 768] :f32]]] k/gelu)
          exec (compile/compile-graph api client graph)]
      (is (some? exec)))))
