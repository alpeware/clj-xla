(ns clj-xla.trace
  "Symbolic execution tracing engine."
  (:require [clj-xla.stablehlo :as shlo]
            [clj-xla.tensor :as t]))

(defn trace-graph
  "Traces execution of function `f` with inputs `invars` to produce an EDN SSA graph map."
  [graph-name invars f]
  (let [eqns-atom (atom [])
        counter-atom (atom 0)
        ctx {:eqns eqns-atom :var-counter counter-atom}
        input-tracers (mapv (fn [[v t]] (t/->Tracer v t)) invars)
        out-tracers (binding [t/*trace-ctx* ctx]
                      (let [res (apply f input-tracers)]
                        (if (vector? res) res [res])))
        outvars (mapv :id out-tracers)
        raw-graph {:name graph-name
                   :invars invars
                   :outvars outvars
                   :eqns @eqns-atom
                   :f f}]
    (shlo/validate-graph raw-graph)))
