(ns clj-xla.sampling-test
  "Unit and generative property tests for model-agnostic logit sampling math."
  (:require [clj-xla.sampling :as sampling]
            [clojure.test :refer [deftest is testing]]
            [clojure.test.check.clojure-test :refer [defspec]]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]))

(defspec prop-apply-temperature 50
  (prop/for-all [temp (gen/choose 1 10)]
                (let [logits [1.0 2.0 3.0 4.0]
                      scaled (sampling/apply-temperature logits (/ temp 10.0))]
                  (and (= (count logits) (count scaled))
                       (every? number? scaled)))))

(defspec prop-apply-top-k 50
  (prop/for-all [k (gen/choose 1 5)]
                (let [logits [10.0 2.0 5.0 1.0 8.0 3.0]
                      filtered (sampling/apply-top-k logits k)]
                  (and (= (count logits) (count filtered))
                       (= k (count (remove #(= % Double/NEGATIVE_INFINITY) filtered)))))))

(deftest sample-logits-test
  (testing "Greedy temperature -> 0 sampling returns argmax token index"
    (let [logits [1.0 10.0 2.0 0.5]
          idx (sampling/sample-logits logits {:temperature 0.0001 :top-k 1})]
      (is (= 1 idx))))
  (testing "Top-K filtering zeroes out non-top-k options"
    (let [logits [100.0 0.0 0.0 0.0]
          idx (sampling/sample-logits logits {:top-k 1})]
      (is (= 0 idx)))))
