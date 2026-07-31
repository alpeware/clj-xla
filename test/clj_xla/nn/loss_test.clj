(ns clj-xla.nn.loss-test
  "Unit and generative tests for loss functions."
  (:require [clj-xla.nn.loss :as loss]
            [clj-xla.tensor :as t :refer [tracer?]]
            [clj-xla.trace :refer [trace-graph]]
            [clojure.test :refer [deftest is testing]]
            [clojure.test.check.clojure-test :refer [defspec]]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]))

(defspec prop-loss-tracing 50
  (prop/for-all [_batch (gen/choose 1 8)]
                (let [g-ce (trace-graph "ce_loss"
                                        [[:logits [:tensor [1 128 50257] :f32]]
                                         [:labels [:tensor [1 128] :i32]]]
                                        (fn [logits labels] (loss/cross-entropy logits labels)))]
                  (and (= "ce_loss" (:name g-ce))
                       (seq (:eqns g-ce))))))

(deftest loss-tracer-test
  (testing "Cross-entropy loss returns Tracer on Tracer inputs"
    (let [logits (t/->Tracer :logits [:tensor [1 128 50257] :f32])
          labels (t/->Tracer :labels [:tensor [1 128] :i32])]
      (is (tracer? (loss/cross-entropy logits labels))))))
