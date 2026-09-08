(ns clj-xla.logic.ast-test
  "Generative property tests for Tensor Logic Hiccup AST schema validation."
  (:require [clj-xla.logic.ast :as ast]
            [clj-xla.logic.generators :as lg]
            [clojure.test.check.clojure-test :refer [defspec]]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]))

(defspec prop-primitive-equations-satisfy-schema
  100
  (prop/for-all [eqn lg/gen-primitive-eqn]
                (ast/valid-node? eqn)))

(defspec prop-binary-contractions-satisfy-schema
  100
  (prop/for-all [eqn lg/gen-binary-contraction]
                (and (ast/valid-node? eqn)
                     (ast/eqn? eqn))))

(defspec prop-while-node-satisfies-schema
  50
  (prop/for-all [out-n (gen/fmap #(keyword (str "out_" %)) (gen/choose 1 100))
                 in-n (gen/fmap #(keyword (str "in_" %)) (gen/choose 1 100))]
                (ast/valid-node? [:while [out-n] [in-n] {:max-iters 10}])))
