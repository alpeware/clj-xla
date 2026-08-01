(ns clj-xla.fetch-pjrt-binaries-test
  "Generative property tests for fetch_pjrt_binaries sources and target resolution."
  (:require [clojure.test :refer [deftest is testing]]
            [clojure.test.check.clojure-test :refer [defspec]]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]
            [scripts.fetch-pjrt-binaries :as fetch]))

(deftest test-sources-structure
  (testing "SOURCES map contains required targets and non-nil attributes"
    (is (contains? fetch/SOURCES :cpu))
    (is (contains? fetch/SOURCES :cuda12))
    (is (contains? fetch/SOURCES :sycl))
    (is (contains? fetch/SOURCES :rocm))
    (doseq [[target config] fetch/SOURCES]
      (is (string? (:url config)) (str "Target " target " missing string url"))
      (is (re-find #"^https?://" (:url config)) (str "Target " target " url must start with http(s)://"))
      (is (string? (:so-name config)) (str "Target " target " missing string so-name"))
      (is (instance? java.util.regex.Pattern (:entry-pattern config)) (str "Target " target " missing entry-pattern regex")))))

(def gen-backend-target
  (gen/elements (conj (vec (keys fetch/SOURCES)) :all)))

(defspec prop-backend-target-resolution
  50
  (prop/for-all [target gen-backend-target]
                (if (= target :all)
                  (= (set (keys fetch/SOURCES)) #{:cpu :cuda12 :sycl :rocm})
                  (some? (get fetch/SOURCES target)))))
