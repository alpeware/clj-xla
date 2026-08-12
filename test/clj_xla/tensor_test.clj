(ns clj-xla.tensor-test
  "Unit and generative tests for clj-xla.tensor shadowed math operators."
  (:require [clj-xla.tensor :as t]
            [clojure.test :refer [deftest is testing]]
            [clojure.test.check.clojure-test :refer [defspec]]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]))

(defspec prop-scalar-auto-lifting 50
  (prop/for-all [c-val gen/double]
                (let [tracer-in (t/->Tracer :x [:tensor [1 10] :f32])
                      res (t/+ tracer-in c-val)]
                  (t/tracer? res))))

(defspec prop-new-tensor-operators-tracing 50
  (prop/for-all [vocab-size (gen/choose 100 1000)
                 hidden-dim (gen/choose 16 128)
                 seq-len (gen/choose 1 32)]
                (binding [t/*trace-ctx* {:var-counter (atom 0) :eqns (atom [])}]
                  (let [wte (t/->Tracer :wte [:tensor [vocab-size hidden-dim] :f32])
                        x (t/->Tracer :x [:tensor [1 seq-len] :i32])
                        g-res (t/gather wte x)
                        wte-t (t/transpose wte [1 0])
                        m-res (t/matmul g-res wte-t)
                        r-res (t/reshape m-res [seq-len vocab-size])
                        t-res (t/transpose r-res [1 0])
                        e-res (t/exp t-res)
                        sum-res (t/reduce-sum e-res :axes [0] :keep-dims true)
                        max-res (t/reduce-max e-res :axes [1] :keep-dims true)]
                    (and (t/tracer? g-res)
                         (= [:tensor [1 seq-len hidden-dim] :f32] (:type g-res))
                         (t/tracer? m-res)
                         (= [:tensor [1 seq-len vocab-size] :f32] (:type m-res))
                         (t/tracer? r-res)
                         (= [:tensor [seq-len vocab-size] :f32] (:type r-res))
                         (t/tracer? t-res)
                         (= [:tensor [vocab-size seq-len] :f32] (:type t-res))
                         (t/tracer? e-res)
                         (= [:tensor [vocab-size seq-len] :f32] (:type e-res))
                         (t/tracer? sum-res)
                         (= [:tensor [1 seq-len] :f32] (:type sum-res))
                         (t/tracer? max-res)
                         (= [:tensor [vocab-size 1] :f32] (:type max-res)))))))

(deftest tracer-pred-test
  (testing "Tracer predicate type checking"
    (is (true? (t/tracer? (t/->Tracer :a [:tensor [1] :f32]))))
    (is (false? (t/tracer? 42)))
    (is (false? (t/tracer? "not-a-tracer")))))

(deftest shadowed-operators-test
  (testing "Shadowed tensor math operators with Tracer inputs"
    (let [x (t/->Tracer :x [:tensor [2 4] :f32])
          y (t/->Tracer :y [:tensor [2 4] :f32])]
      (is (t/tracer? (t/+ x y)))
      (is (t/tracer? (t/- x y)))
      (is (t/tracer? (t/* x y)))
      (is (t/tracer? (t// x y)))
      (is (t/tracer? (t/pow x 2.0)))
      (is (t/tracer? (t/tanh x)))
      (is (t/tracer? (t/sqrt x))))))

(deftest argmax-tracing-test
  (testing "Argmax tracer produces correct tensor shape and int32 type"
    (binding [t/*trace-ctx* {:var-counter (atom 0) :eqns (atom [])}]
      (let [logits (t/->Tracer :logits [:tensor [1 1 262144] :f32])
            tok-id (t/argmax logits :axis -1)]
        (is (t/tracer? tok-id))
        (is (= [:tensor [1 1] :i32] (:type tok-id)))))))
