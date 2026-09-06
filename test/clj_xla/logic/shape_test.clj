(ns clj-xla.logic.shape-test
  "Generative property and unit tests for Tensor Logic shape unification."
  (:require [clj-xla.logic.shape :as shape]
            [clojure.test :refer [deftest is]]))

(deftest test-unify-shapes-standard-matmul
  (let [in-shapes {:x [2 128 768] :w [768 3072]}
        eqn [:= [:y :b :p :dff] [:x :b :p :d] [:w :d :dff]]
        resolved (shape/unify-shapes in-shapes [eqn])]
    (is (= [2 128 3072] (get resolved :y)))))

(deftest test-unify-shapes-detects-mismatch
  (let [in-shapes {:x [2 128 768] :w [512 3072]}
        eqn [:= [:y :b :p :dff] [:x :b :p :d] [:w :d :dff]]]
    (is (thrown? Exception (shape/unify-shapes in-shapes [eqn])))))
