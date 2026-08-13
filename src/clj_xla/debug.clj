(ns clj-xla.debug
  "Debugging, NaN checkify assertions, and XLA metadata annotation utilities."
  (:require [clj-xla.tensor :as t]))

(def ^:dynamic *xla-metadata* nil)

(defmacro with-xla-metadata
  "Dynamic scope macro attaching XLA location metadata (op-name, source-file, source-line)
   to all equation nodes traced within body."
  [meta-map & body]
  `(let [meta# ~meta-map
         before-idx# (if t/*trace-ctx* (count @(:eqns t/*trace-ctx*)) 0)
         res# (do ~@body)]
     (when t/*trace-ctx*
       (let [eqns# @(:eqns t/*trace-ctx*)
             after-idx# (count eqns#)]
         (dotimes [i# (- after-idx# before-idx#)]
           (let [idx# (+ before-idx# i#)
                 eqn# (nth eqns# idx#)
                 updated# (assoc eqn# :metadata (merge (:metadata eqn#) meta#))]
             (swap! (:eqns t/*trace-ctx*) assoc idx# updated#)))))
     res#))

(defn check-non-nan
  "Emits a functional `checkify` NaN assertion node into the trace graph context.
   If `val` evaluates to NaN during XLA execution, raises `msg`."
  ([val] (check-non-nan val "NaN detected in intermediate tensor computation"))
  ([val msg]
   (when t/*trace-ctx*
     (let [chk-id (t/gen-var-id! "chk_nan")
           eqn {:op :debug/check-non-nan
                :invars [(if (t/tracer? val) (:id val) val)]
                :outvars [chk-id]
                :message msg
                :metadata *xla-metadata*}]
       (swap! (:eqns t/*trace-ctx*) conj eqn)))
   val))

(defn check-non-inf
  "Emits a functional `checkify` Infinity assertion node into the trace graph context.
   If `val` evaluates to Infinity during XLA execution, raises `msg`."
  ([val] (check-non-inf val "Infinity detected in intermediate tensor computation"))
  ([val msg]
   (when t/*trace-ctx*
     (let [chk-id (t/gen-var-id! "chk_inf")
           eqn {:op :debug/check-non-inf
                :invars [(if (t/tracer? val) (:id val) val)]
                :outvars [chk-id]
                :message msg
                :metadata *xla-metadata*}]
       (swap! (:eqns t/*trace-ctx*) conj eqn)))
   val))

(defn debug-print
  "Emits a compiled debug print equation into the trace graph, logging `label` and `val` at runtime."
  [val label]
  (when t/*trace-ctx*
    (let [dbg-id (t/gen-var-id! "dbg_print")
          eqn {:op :debug/print
               :invars [(if (t/tracer? val) (:id val) val)]
               :outvars [dbg-id]
               :label label
               :metadata *xla-metadata*}]
      (swap! (:eqns t/*trace-ctx*) conj eqn)))
  val)
