(ns clj-xla.pjrt.version-test
  "Generative property tests for PJRT versioning, compatibility validation, and driver probing."
  (:require [clj-xla.pjrt.version :as v]
            [clojure.test :refer [deftest is testing]]
            [clojure.test.check.clojure-test :refer [defspec]]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]))

(def gen-version-pair
  (gen/tuple gen/small-integer gen/small-integer))

(defspec prop-compatibility-validation-invariant
  100
  (prop/for-all [[major minor] gen-version-pair
                 min-minor gen/small-integer]
                (let [res (v/validate-version-compatibility major minor min-minor)
                      expected-compat (and (= major 0) (>= minor min-minor))]
                  (and (boolean? (:compatible? res))
                       (= expected-compat (:compatible? res))
                       (= major (:major res))
                       (= minor (:minor res))
                       (string? (:reason res))))))

(defspec prop-plugin-attributes-parsing
  50
  (prop/for-all [attrs (gen/map (gen/not-empty gen/string-alphanumeric)
                                (gen/one-of [(gen/not-empty gen/string-alphanumeric) gen/small-integer]))]
                (let [parsed (v/parse-plugin-attributes attrs)]
                  (and (map? parsed)
                       (contains? parsed :raw-attributes)
                       (or (nil? (:driver-version parsed)) (string? (:driver-version parsed)))))))

(deftest test-system-driver-probing
  (testing "System driver probe returns structured non-nil diagnostic map"
    (let [probe (v/probe-system-driver)]
      (is (map? probe))
      (is (set? (:detected-backends probe)))
      (is (map? (:details probe)))
      (is (contains? (:detected-backends probe) :cpu) "CPU backend must always be detected as fallback"))))
