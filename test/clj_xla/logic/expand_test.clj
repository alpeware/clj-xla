(ns clj-xla.logic.expand-test
  "Generative property tests for Tensor Logic AST recursive expansion."
  (:require [clj-xla.logic.ast :as ast]
            [clj-xla.logic.expand :as expand]
            [clj-xla.logic.generators :as lg]
            [clojure.test :refer [deftest is]]
            [clojure.test.check.clojure-test :refer [defspec]]
            [clojure.test.check.properties :as prop]))

(defspec prop-primitive-expansion-idempotence
  100
  (prop/for-all [eqn lg/gen-binary-contraction]
                (let [ctx {}
                      exp1 (expand/expand-ast ctx [eqn])
                      exp2 (expand/expand-ast ctx exp1)]
                  (= exp1 exp2))))

(deftest test-block-container-expansion
  (let [ast [:block {:name :layer-0}
             [:= [:x1 :b :p :d] [:x0 :b :p :d] [:w1 :d :d]]
             [:= [:x2 :b :p :d] [:x1 :b :p :d] [:w2 :d :d]]]
        expanded (expand/expand-ast {} ast)]
    (is (= 2 (count expanded)))
    (is (every? ast/eqn? expanded))
    (is (= :x1 (first (second (first expanded)))))
    (is (= :x2 (first (second (second expanded)))))))

(deftest test-nested-block-expansion
  (let [ast [:block {:name :root}
             [:block {:name :child-1}
              [:= [:a :i] [:b :i]]]
             [:block {:name :child-2}
              [:= [:c :i] [:d :i]]]]
        expanded (expand/expand-ast {} ast)]
    (is (= 2 (count expanded)))
    (is (every? ast/eqn? expanded))))

(deftest test-multi-term-pairwise-decomposition
  (let [ast [:= [:out :i :l]
             [:a :i :j]
             [:b :j :k]
             [:c :k :l]]
        expanded (expand/expand-ast {} ast)]
    ;; 3 terms should decompose into 2 binary contraction equations
    (is (= 2 (count expanded)))
    (is (every? ast/eqn? expanded))
    ;; The final equation produces :out
    (is (= :out (first (second (last expanded)))))
    ;; Each equation has at most 2 body terms
    (is (every? (fn [eqn] (<= (count (ast/body-terms eqn)) 2)) expanded))))
