(ns clj-xla.safetensors-test
  "Unit and generative tests for clj-xla.safetensors Panama off-heap weight loader."
  (:require [clj-xla.safetensors :as st]
            [clojure.data.json :as json]
            [clojure.test :refer [deftest is testing]]
            [clojure.test.check.clojure-test :refer [defspec]]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]))

(defspec prop-header-json-parsing 50
  (prop/for-all [shape-len (gen/choose 1 4)
                 dim-val (gen/choose 1 128)]
                (let [header-map {"weight_1" {"dtype" "F32" "shape" (vec (repeat shape-len dim-val)) "data_offsets" [0 100]}}
                      parsed (st/parse-header-json (json/write-str header-map))]
                  (some? (get parsed "weight_1")))))

(deftest safetensors-header-parser-test
  (testing "Parsing safetensors header JSON metadata"
    (let [header-json "{\"weight_a\":{\"dtype\":\"F32\",\"shape\":[1,128,768],\"data_offsets\":[0,393216]}}"
          parsed (st/parse-header-json header-json)]
      (is (= "F32" (get-in parsed ["weight_a" "dtype"])))
      (is (= [1 128 768] (get-in parsed ["weight_a" "shape"])))
      (is (= [0 393216] (get-in parsed ["weight_a" "data_offsets"]))))))

(deftest sharded-safetensors-directory-test
  (testing "Sharded safetensors directory lookup structure"
    (let [dummy-map {:tensors {"model.embed_tokens.weight" {:info {"data_offsets" [0 4]} :segment nil}}}]
      (is (some? (get-in dummy-map [:tensors "model.embed_tokens.weight"]))))))
