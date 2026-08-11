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
   and returns the compiled PjRtLoadedExecutable handle map."
  [api-ctx client graph]
  (let [mlir-text (shlo/graph->mlir-text graph)
        hash-key [(or client (:client api-ctx)) (sha256-hash mlir-text)]]
    (if-let [cached-exec (get @exec-cache hash-key)]
      (assoc cached-exec :f (:f graph) :graph graph)
      (let [exec (pjrt/compile-mlir api-ctx client mlir-text)
            exec-obj {:handle exec :hash hash-key :f (:f graph) :graph graph}]
        (swap! exec-cache assoc hash-key exec-obj)
        exec-obj))))

(defn clear-cache!
  "Clears the in-memory compilation cache."
  []
  (reset! exec-cache {}))
