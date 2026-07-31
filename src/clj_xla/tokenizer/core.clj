(ns clj-xla.tokenizer.core
  "Factory dispatcher for Tokenizers based on file paths and formats."
  (:require [clj-xla.tokenizer.bpe :as bpe]
            [clj-xla.tokenizer.hf-json :as hf]
            [clj-xla.tokenizer.sentencepiece :as sp]
            [clojure.java.io :as io]
            [clojure.string :as str]))

(defn from-file
  "Auto-detects tokenizer format from file path or directory and returns a Tokenizer instance."
  [path]
  (let [file (io/file path)]
    (cond
      (and (.isDirectory file) (.exists (io/file file "tokenizer.json")))
      (hf/load-hf-json-tokenizer (.getAbsolutePath (io/file file "tokenizer.json")))

      (and (.isDirectory file) (.exists (io/file file "vocab.json")) (.exists (io/file file "merges.txt")))
      (bpe/load-bpe-tokenizer (.getAbsolutePath (io/file file "vocab.json"))
                              (.getAbsolutePath (io/file file "merges.txt")))

      (str/ends-with? path "tokenizer.json")
      (hf/load-hf-json-tokenizer path)

      (str/ends-with? path "tokenizer.model")
      (sp/load-sentencepiece-tokenizer path)

      :else
      (throw (ex-info "Unable to auto-detect tokenizer format from path" {:path path})))))
