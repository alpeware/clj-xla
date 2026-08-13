(ns clj-xla.agent-test
  "Unit and generative tests for scripts.gemma4-agent SCI tool execution and turn formatting."
  (:require [clojure.string :as str]
            [clojure.test :refer [deftest is testing]]
            [clojure.test.check.clojure-test :refer [defspec]]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]
            [sci.core :as sci]))

(defn extract-clojure-code-blocks
  "Extracts all ```clojure ... ``` code block strings from text, including unclosed blocks up to EOF."
  [text]
  (let [pattern #"(?s)```(?:clojure|clj)\s*\n(.*?)(?:```|$)"]
    (mapv str/trim (filter #(seq (str/trim %)) (mapv second (re-seq pattern text))))))

(defn eval-sci-code
  "Evaluates `code-str` in a safe SCI sandbox context with agent helper bindings."
  [code-str]
  (let [env (sci/init {:bindings {'println println
                                  'str str
                                  'range range
                                  'mapv mapv
                                  'reduce reduce
                                  'filter filter}})]
    (try
      (let [out (binding [*out* (java.io.StringWriter.)]
                  (let [res (sci/eval-string* env code-str)
                        printed (str *out*)]
                    (if (seq printed)
                      (str printed "\n=> " (pr-str res))
                      (pr-str res))))]
        {:status :success :result out})
      (catch Exception e
        {:status :error :result (str "Evaluation Error: " (.getMessage e))}))))

(deftest test-extract-clojure-code-blocks
  (testing "Extracting single and multiple Clojure code blocks from model generation"
    (let [text "I will calculate the sum of squares.\n```clojure\n(reduce + (map #(Math/pow % 2) (range 1 5)))\n```\nDone."
          blocks (extract-clojure-code-blocks text)]
      (is (= 1 (count blocks)))
      (is (clojure.string/includes? (first blocks) "reduce +")))))

(deftest test-eval-sci-code-success
  (testing "Evaluating math expressions in SCI sandbox"
    (let [res (eval-sci-code "(reduce + (range 10))")]
      (is (= :success (:status res)))
      (is (clojure.string/includes? (:result res) "45")))))

(deftest test-eval-sci-code-error-handling
  (testing "Handling runtime exceptions inside SCI sandbox"
    (let [res (eval-sci-code "(/ 1 0)")]
      (is (= :error (:status res)))
      (is (clojure.string/includes? (:result res) "Evaluation Error")))))

(defspec prop-sci-arithmetic-eval-invariant
  50
  (prop/for-all [a (gen/choose 1 1000)
                 b (gen/choose 1 1000)]
                (let [code (format "(+ %d %d)" a b)
                      res (eval-sci-code code)]
                  (and (= :success (:status res))
                       (= (str (+ a b)) (str/trim (:result res)))))))
