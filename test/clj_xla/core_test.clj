(ns clj-xla.core-test
  "Unit and generative property tests for dynamic multi-backend PJRT plugin initialization in clj-xla.core."
  (:require [clj-xla.core :as xla]
            [clojure.test :refer [deftest is testing]]
            [clojure.test.check.clojure-test :refer [defspec]]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]))

(deftest test-backend-library-map-structure
  (testing "BACKEND-LIBRARY-MAP contains all supported hardware targets"
    (is (contains? xla/BACKEND-LIBRARY-MAP :cpu))
    (is (contains? xla/BACKEND-LIBRARY-MAP :sycl))
    (is (contains? xla/BACKEND-LIBRARY-MAP :rocm))
    (is (contains? xla/BACKEND-LIBRARY-MAP :cuda12))
    (doseq [[target config] xla/BACKEND-LIBRARY-MAP]
      (is (string? (:default config)) (str "Target " target " missing string default"))
      (is (re-find #"^bin/libpjrt_" (:default config)) (str "Target " target " default must start with bin/libpjrt_"))
      (is (string? (:env config)) (str "Target " target " missing string env"))
      (is (re-find #"^PJRT_.*_PATH$" (:env config)) (str "Target " target " env must match PJRT_*_PATH")))))

(def gen-target-keyword
  (gen/elements [:cpu :sycl :rocm :cuda12]))

(defspec prop-backend-resolution-invariants
  50
  (prop/for-all [target gen-target-keyword]
                (let [config (get xla/BACKEND-LIBRARY-MAP target)]
                  (and (some? config)
                       (string? (:default config))
                       (string? (:env config))))))
