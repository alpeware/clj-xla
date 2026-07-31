(ns clj-xla.compile
  (:require [clj-xla.pjrt :as pjrt]
            [clj-xla.stablehlo :as shlo])
  (:import [java.security MessageDigest]))

(defonce ^:private exec-cache (atom {}))

(defn- sha256-hash [^String s]
  (let [digest (MessageDigest/getInstance "SHA-256")
        bytes (.digest digest (.getBytes s "UTF-8"))]
    (apply str (map #(format "%02x" %) bytes))))

(defn compile-graph
  "Serializes `graph` to StableHLO MLIR text, checks the in-memory SHA-256 cache,
   and returns the compiled PjRtLoadedExecutable handle."
  [api-ctx client graph]
  (let [mlir-text (shlo/graph->mlir-text graph)
        hash-key (sha256-hash mlir-text)]
    (if-let [cached-exec (get @exec-cache hash-key)]
      cached-exec
      (let [exec (pjrt/compile-mlir api-ctx client mlir-text)]
        (swap! exec-cache assoc hash-key exec)
        exec))))

(defn clear-cache!
  "Clears the in-memory compilation cache."
  []
  (reset! exec-cache {}))
