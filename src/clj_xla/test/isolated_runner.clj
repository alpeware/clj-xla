(ns clj-xla.test.isolated-runner
  "Process-isolated test runner harness for executing PJRT device tests in separate JVM subprocesses."
  (:require [clojure.edn :as edn]
            [clojure.java.shell :refer [sh]]))

(defn- parse-summary-map
  "Parses the printed summary EDN map from worker stdout."
  [stdout]
  (when stdout
    (when-let [match (re-find #"SUMMARY:\s*(\{:test[\s\S]+?\})" stdout)]
      (try
        (edn/read-string (second match))
        (catch Exception _ nil)))))

(defn run-isolated-test
  "Executes test namespace `test-ns-sym` in a clean, process-isolated JVM worker.
   Prevents native C/C++ segfaults or signal aborts from taking down the main test suite runner."
  ([test-ns-sym] (run-isolated-test test-ns-sym {}))
  ([test-ns-sym env-map]
   (let [java-bin (str (System/getProperty "java.home") "/bin/java")
         cp (System/getProperty "java.class.path")
         expr (format "(require '%s) (let [res (clojure.test/run-tests '%s)] (println \"SUMMARY:\" (pr-str res)))"
                      (name test-ns-sym) (name test-ns-sym))
         cmd [java-bin "--enable-native-access=ALL-UNNAMED" "-cp" cp "clojure.main" "-e" expr]
         merged-env (merge (into {} (System/getenv)) (or env-map {}))
         res (apply sh (concat cmd [:env merged-env]))
         exit-code (:exit res)
         stdout (:out res)
         stderr (:err res)
         summary (parse-summary-map stdout)
         passed? (and (zero? exit-code)
                      (some? summary)
                      (zero? (get summary :fail 0))
                      (zero? (get summary :error 0)))]
     {:namespace test-ns-sym
      :status (if passed? :pass :failed-or-crashed)
      :exit exit-code
      :summary summary
      :stdout stdout
      :stderr stderr})))

(defn run-isolated-suite
  "Executes a collection of test namespace symbols sequentially in isolated JVM subprocesses."
  ([namespaces] (run-isolated-suite namespaces {}))
  ([namespaces env-map]
   (let [results (mapv #(run-isolated-test % env-map) namespaces)
         passed (filter #(= (:status %) :pass) results)
         crashed (filter #(not= (:status %) :pass) results)]
     {:total-namespaces (count namespaces)
      :passed-namespaces (count passed)
      :crashed-namespaces (count crashed)
      :details results})))
