(ns clj-xla.generation-test
  "Unit and generative tests for autoregressive text generation strategy."
  (:require [clj-xla.generation.autoregressive :as ar]
            [clj-xla.tokenizer.bpe :as bpe]
            [clj-xla.tokenizer.protocol :refer [encode]]
            [clojure.test :refer [deftest is testing]]
            [clojure.test.check.clojure-test :refer [defspec]]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]))

(deftest autoregressive-generate-mock-test
  (testing "Autoregressive generation step loop with dummy step function"
    (let [tokenizer (bpe/load-bpe-tokenizer ".models/gpt2/vocab.json" ".models/gpt2/merges.txt")
          prompt-ids (encode tokenizer "Hello world")
          step-fn (fn [_context-ids]
                    (vec (concat (repeat 50255 0.0) [100.0])))
          gen-ids (ar/generate-tokens step-fn prompt-ids {:max-new-tokens 5 :eos-token-id 50256})
          out-text (clj-xla.tokenizer.protocol/decode tokenizer gen-ids)]
      (is (vector? gen-ids))
      (is (> (count gen-ids) (count prompt-ids)))
      (is (string? out-text)))))

(defspec prop-autoregressive-max-tokens 20
  (prop/for-all [max-tokens (gen/choose 1 5)]
                (let [prompt-ids [15496 995]
                      step-fn (fn [_ctx] [1.0 2.0 3.0 4.0])
                      gen-ids (ar/generate-tokens step-fn prompt-ids {:max-new-tokens max-tokens :eos-token-id 9999})]
                  (= (+ (count prompt-ids) max-tokens) (count gen-ids)))))
