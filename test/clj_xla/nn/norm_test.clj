(ns clj-xla.nn.norm-test
  "Unit and generative tests for normalization layers."
  (:require [clj-xla.nn.norm :as norm]
            [clj-xla.tensor :as t :refer [tracer?]]
            [clj-xla.trace :refer [trace-graph]]
            [clojure.test :refer [deftest is testing]]
            [clojure.test.check.clojure-test :refer [defspec]]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]))

(defspec prop-norm-tracing 50
  (prop/for-all [_seq-len (gen/choose 1 64)]
                (let [g-ln (trace-graph "ln_test"
                                        [[:x [:tensor [1 128 768] :f32]]
                                         [:gamma [:tensor [768] :f32]]
                                         [:beta [:tensor [768] :f32]]]
                                        (fn [x g b] (norm/layer-norm x g b)))
                      g-rms (trace-graph "rms_test"
                                         [[:x [:tensor [1 128 768] :f32]]
                                          [:weight [:tensor [768] :f32]]]
                                         (fn [x w] (norm/rms-norm x w)))]
                  (and (= "ln_test" (:name g-ln))
                       (= "rms_test" (:name g-rms))
                       (seq (:eqns g-ln))
                       (seq (:eqns g-rms))))))

(deftest norm-tracer-test
  (testing "Normalization layers return Tracer on Tracer input"
    (let [x (t/->Tracer :x [:tensor [1 128 768] :f32])
          g (t/->Tracer :g [:tensor [768] :f32])
          b (t/->Tracer :b [:tensor [768] :f32])]
      (is (tracer? (norm/layer-norm x g b)))
      (is (tracer? (norm/rms-norm x g))))))
