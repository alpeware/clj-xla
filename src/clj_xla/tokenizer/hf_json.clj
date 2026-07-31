(ns clj-xla.tokenizer.hf-json
  "HuggingFace fast tokenizer.json loader."
  (:require [clj-xla.tokenizer.bpe :as bpe]
            [clj-xla.tokenizer.protocol :refer [Tokenizer]]
            [clojure.data.json :as json]
            [clojure.string :as str]))

(defrecord HFJsonTokenizer [vocab encoder bpe-ranks bos-token-id eos-token-id]
  Tokenizer
  (encode [this text]
    (clj-xla.tokenizer.protocol/encode this text false))
  (encode [_this text _add-special-tokens?]
    (let [bpe-tok (bpe/->BPETokenizer vocab encoder bpe-ranks bos-token-id eos-token-id)]
      (clj-xla.tokenizer.protocol/encode bpe-tok text _add-special-tokens?)))

  (decode [this token-ids]
    (clj-xla.tokenizer.protocol/decode this token-ids true))
  (decode [_this token-ids _skip-special-tokens?]
    (let [bpe-tok (bpe/->BPETokenizer vocab encoder bpe-ranks bos-token-id eos-token-id)]
      (clj-xla.tokenizer.protocol/decode bpe-tok token-ids _skip-special-tokens?)))

  (bos-id [_this] bos-token-id)
  (eos-id [_this] eos-token-id))

(defn load-hf-json-tokenizer
  "Loads HuggingFace `tokenizer.json` file keeping raw string tokens intact."
  [json-path]
  (let [data (json/read-str (slurp json-path))
        vocab-map (get-in data ["model" "vocab"])
        merges (get-in data ["model" "merges"])
        encoder (into {} (map (fn [[k v]] [k (int v)]) vocab-map))
        vocab (into {} (map (fn [[k v]] [v k]) encoder))
        merge-pairs (for [item merges
                          :let [parts (str/split item #" ")]
                          :when (= 2 (count parts))]
                      [(first parts) (second parts)])
        bpe-ranks (into {} (map-indexed (fn [i pair] [pair i]) merge-pairs))
        eos-id (get encoder "<|endoftext|>" 50256)]
    (->HFJsonTokenizer vocab encoder bpe-ranks eos-id eos-id)))
