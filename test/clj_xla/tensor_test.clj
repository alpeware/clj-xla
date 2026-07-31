(ns clj-xla.tensor-test
  "Unit and generative tests for clj-xla.tensor shadowed math operators."
  (:refer-clojure :exclude [+ * - /])
  (:require [clj-xla.tensor :as t :refer [+ * - / pow sqrt tanh tracer?]]
            [clojure.test :refer [deftest is testing]]
            [clojure.test.check.clojure-test :refer [defspec]]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]))

(defspec prop-scalar-auto-lifting 50
  (prop/for-all [c-val gen/double]
                (let [tracer-in (t/->Tracer :x [:tensor [1 10] :f32])
                      res (+ tracer-in c-val)]
                  (tracer? res))))

(deftest tracer-pred-test
  (testing "Tracer predicate type checking"
    (is (true? (tracer? (t/->Tracer :a [:tensor [1] :f32]))))
    (is (false? (tracer? 42)))
    (is (false? (tracer? "not-a-tracer")))))

(deftest shadowed-operators-test
  (testing "Shadowed tensor math operators with Tracer inputs"
    (let [x (t/->Tracer :x [:tensor [2 4] :f32])
          y (t/->Tracer :y [:tensor [2 4] :f32])]
      (is (tracer? (+ x y)))
      (is (tracer? (- x y)))
      (is (tracer? (* x y)))
      (is (tracer? (/ x y)))
      (is (tracer? (pow x 2.0)))
      (is (tracer? (tanh x)))
      (is (tracer? (sqrt x))))))
