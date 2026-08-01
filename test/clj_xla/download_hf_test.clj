(ns clj-xla.download-hf-test
  "Unit and generative tests for HuggingFace download utilities."
  (:require [clojure.test :refer [deftest is testing]]
            [clojure.test.check.clojure-test :refer [defspec]]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]
            [scripts.download-hf :as dl]))

(defspec prop-filter-relevant-files 50
  (prop/for-all [ext (gen/elements ["safetensors" "json" "txt" "model" "bin" "md" "py"])]
                (let [filename (str "model." ext)
                      filtered (dl/filter-relevant-files [filename])]
                  (if (contains? #{"safetensors" "json" "txt" "model"} ext)
                    (= [filename] filtered)
                    (empty? filtered)))))

(deftest parse-manifest-rpaths-test
  (testing "Extracts rpath/rfilename from manifest response string"
    (let [json-body "{\"siblings\":[{\"rfilename\":\"config.json\"},{\"rpath\":\"model.safetensors\"}]}"
          rpaths (dl/parse-manifest-rpaths json-body)]
      (is (= ["config.json" "model.safetensors"] rpaths)))))
